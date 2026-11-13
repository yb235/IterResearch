# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/eval_interval.py
"""Sub-task N: score the lower-claim variants (lower_rule.py) on the no-filing backtests.

Inputs (reports/, written by scripts/forecast_interval.py and scripts/forecast_ml_fixed_interval.py):
  forecast_interval_<model>_<variant>_<rule>_2026-11-13.csv   A, B, E, G, R (run_backtest), bayes
  forecast_mlfixed_<variant>_<rule>_2026-11-13.csv             M1 / M2 (pooled, own), SS claims-only path
Naive = the last point claim (backtest._naive, from A's baseline run; it reads point claims only, so
it is the same under every variant). Errors are log(actual / forecast): + = forecast too low.

Target sets, per vintage rule:
  all      targets every model x variant in the table forecast (common targets)
  post-t0  of those, targets after the last anchor (point claim) public at the vintage, the
           definition of reports/forecast_mlfixed_2026-11-10.md (ml_fixed spec item 7): the flag of
           forecast_mlfixed_<rule>_2026-11-10.csv where it exists, else recomputed with the same rule
           (anchors at the crossing position profiled at the vintage, one-sided baseline rule)
  H2-2026  rev_2026q3, rr_2026_09_s1a, rr_2026_10_s1a (forecast_eval.H2_TARGETS)
Paired comparisons: forecast_eval.paired (difference in |log error|, 10000 bootstrap draws, 90% interval).

Outputs: reports/eval_interval_2026-11-13.md, eval_interval_long_2026-11-13.csv,
eval_interval_summary_2026-11-13.csv, eval_interval_paired_2026-11-13.csv,
forecast_vs_actual_interval_2026-11-13.png.
"""
import warnings; warnings.simplefilter("ignore")
import glob, json, math, os, re
import numpy as np
import pandas as pd

from runrate_nowcast import config, data, forecast_eval, lower_rule
from runrate_nowcast import ml_fixed as mf
from runrate_nowcast import panel_model as pm

ASOF = "2026-11-13"
REP = config.OUTPUT_DIR
RULES = ("publication", "period_end")
VARIANTS = lower_rule.ORDER
PRIMARY = lower_rule.PRIMARY
H2 = forecast_eval.H2_TARGETS
NAIVE = "N naive last claim"
LABELS = {"A": "A claims only", "B": "B + indicators", "E": "E OpenRouter tokens x price",
          "G": "G GitHub new-repo share", "R": "R Ramp adoption", "bayes": "Bayes + indicators"}
ML = {("M1", "pooled"): "M1 (pooled)", ("M2", "pooled"): "M2 (pooled)", ("M1", "own"): "M1 (own)",
      ("M2", "own"): "M2 (own)", (mf.SS_NAME, "own"): "SS claims-only path (ml_fixed)"}
ORDER = [LABELS[k] for k in ("A", "B", "E", "G", "R", "bayes")] + list(ML.values())
STATE_SPACE = [LABELS[k] for k in ("A", "B", "E", "G", "R", "bayes")]


# --------------------------------------------------------------------------- loading
def load_long() -> pd.DataFrame:
    rows = []
    pat = re.compile(rf"forecast_interval_(?P<m>[A-Za-z]+)_(?P<v>[a-z0-9]+)_(?P<r>publication|period_end)_{ASOF}\.csv$")
    for f in sorted(glob.glob(os.path.join(REP, f"forecast_interval_*_{ASOF}.csv"))):
        g = pat.search(os.path.basename(f))
        if not g:
            continue
        m, v, rule = g["m"], g["v"], g["r"]
        d = pd.read_csv(f)
        if m == "bayes":
            d = d[d.method == "laplace_mixture"]
            for r in d.itertuples():
                rr = r.metric == "run_rate"
                fc = math.exp(r.forecast_mu) if rr else r.forecast_mu
                lo = math.exp(r.forecast_mu - 1.2816 * r.forecast_sd) if rr else r.forecast_mu - 1.2816 * r.forecast_sd
                hi = math.exp(r.forecast_mu + 1.2816 * r.forecast_sd) if rr else r.forecast_mu + 1.2816 * r.forecast_sd
                rows.append(dict(rule=rule, model=LABELS[m], variant=v, target=r.target, metric=r.metric,
                                 vintage=r.vintage, actual=r.actual, forecast=fc, q10=lo, q90=hi,
                                 log_err=r.log_err, z=r.z, in80=bool(r.in80), in50=bool(r.in50)))
            continue
        for r in d.itertuples():
            rows.append(dict(rule=rule, model=LABELS[m], variant=v, target=r.target, metric=r.metric,
                             vintage=r.vintage, actual=r.actual, forecast=r.forecast, q10=r.q10, q90=r.q90,
                             log_err=r.log_err_model, z=r.z, in80=bool(r.in80), in50=bool(r.in50)))
            if m == "A" and v == "baseline" and pd.notna(r.log_err_naive):
                rows.append(dict(rule=rule, model=NAIVE, variant="-", target=r.target, metric=r.metric,
                                 vintage=r.vintage, actual=r.actual, forecast=r.naive, q10=np.nan, q90=np.nan,
                                 log_err=r.log_err_naive, z=np.nan, in80=np.nan, in50=np.nan))
    pat = re.compile(rf"forecast_mlfixed_(?P<v>[a-z0-9]+)_(?P<r>publication|period_end)_{ASOF}\.csv$")
    files = [(f, pat.search(os.path.basename(f))) for f in sorted(glob.glob(os.path.join(REP, f"forecast_mlfixed_*_{ASOF}.csv")))]
    files = [(f, {"v": g["v"], "r": g["r"]}) for f, g in files if g]
    for rule in RULES:      # no fresh baseline run: the 2026-11-10 report (same code path with the rule off)
        if files and not any(g["v"] == "baseline" and g["r"] == rule for _, g in files):
            files.append((os.path.join(REP, f"forecast_mlfixed_{rule}_2026-11-10.csv"), {"v": "baseline", "r": rule}))
    for f, g in files:
        d = pd.read_csv(f)
        d = d[(d.company == "anthropic")]
        for r in d.itertuples():
            key = (r.model, r.pool) if r.model != mf.SS_NAME else (r.model, "own")
            if key not in ML or pd.isna(r.log_err):
                continue
            rows.append(dict(rule=g["r"], model=ML[key], variant=g["v"], target=r.target, metric=r.metric,
                             vintage=r.vintage, actual=r.actual, forecast=r.forecast, q10=np.nan, q90=np.nan,
                             log_err=r.log_err, z=np.nan, in80=np.nan, in50=np.nan, post_t0_ml=bool(r.post_t0)))
    return pd.DataFrame(rows)


def openai_long() -> pd.DataFrame:
    """ml_fixed OpenAI targets (second test set), every variant run."""
    out = []
    for f in sorted(glob.glob(os.path.join(REP, f"forecast_mlfixed_*_{ASOF}.csv"))):
        g = re.search(rf"forecast_mlfixed_([a-z0-9]+)_(publication|period_end)_{ASOF}", f)
        d = pd.read_csv(f)
        d = d[(d.company == "openai") & d.model.isin(["M1", "M2", mf.SS_NAME, mf.NAIVE_NAME])
              & ((d.pool == "pooled") | d.model.isin([mf.SS_NAME, mf.NAIVE_NAME]))]
        out.append(d.assign(variant=g.group(1), rule=g.group(2)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------- post-t0
def post_t0_flags(targets: pd.DataFrame) -> dict:
    """{(rule, target): bool}. forecast_mlfixed_<rule>_2026-11-10.csv's flag (baseline rule), else recomputed."""
    flags = {}
    for rule in RULES:
        f = os.path.join(REP, f"forecast_mlfixed_{rule}_2026-11-10.csv")
        if os.path.exists(f):
            z = pd.read_csv(f)
            z = z[(z.company == "anthropic") & (z.model == mf.SS_NAME)]
            flags.update({(rule, t): bool(p) for t, p in zip(z.target, z.post_t0)})
    claims = pm.load_claims("anthropic")
    table = data.load()
    recomputed = {}
    with lower_rule.lower_rule("baseline"):
        for r in targets.drop_duplicates(["rule", "target"]).itertuples():
            row = table[table.obs_id == r.target].iloc[0]
            v = pd.Timestamp(r.vintage)
            rho = mf.crossing_rho("anthropic", claims, v, exclude_origin=row.origin_id)
            sel = pm.public_claims(data.with_crossing_position(claims, rho), v, row.origin_id, start=config.MODEL_START)
            anch = pm.anchors(sel)
            anch = anch[anch.index >= pd.Timestamp(config.MODEL_START)]
            if not len(anch):
                recomputed[(r.rule, r.target)] = True
                continue
            t0 = anch.index.max()
            if row.metric == "revenue":
                last = pd.Timestamp(row.period_end)
            else:
                d = data.with_crossing_position(table, rho)
                last = pd.Timestamp(data.effective_date(d[d.obs_id == r.target].iloc[0]))
            recomputed[(r.rule, r.target)] = bool(last > t0)
    mism = [k for k in flags if k in recomputed and flags[k] != recomputed[k]]
    for k, v in recomputed.items():
        flags.setdefault(k, v)
    return flags, mism


# --------------------------------------------------------------------------- tables
def common(d: pd.DataFrame) -> pd.DataFrame:
    """Rows whose (rule, target) every (model, variant) present in that rule forecast."""
    out = []
    for rule, g in d.groupby("rule"):
        combos = g[["model", "variant"]].drop_duplicates()
        n = len(combos)
        cnt = g.groupby("target").apply(lambda h: len(h[["model", "variant"]].drop_duplicates()))
        out.append(g[g.target.isin(cnt[cnt == n].index)])
    return pd.concat(out, ignore_index=True)


def summary(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rule, model, variant), h in c.groupby(["rule", "model", "variant"]):
        post, h2 = h[h.post_t0], h[h.target.isin(H2)]
        cov = h.in80.dropna()
        rows.append({"rule": rule, "model": model, "variant": variant,
                     "n_all": len(h), "mae_all": h.log_err.abs().mean(), "bias_all": h.log_err.mean(),
                     "n_post": len(post), "mae_post": post.log_err.abs().mean(), "bias_post": post.log_err.mean(),
                     "n_h2": len(h2), "mae_h2": h2.log_err.abs().mean(),
                     "cov80": cov.astype(float).mean() if len(cov) else np.nan,
                     "cov50": h.in50.dropna().astype(float).mean() if len(cov) else np.nan,
                     "cov80_post": post.in80.dropna().astype(float).mean() if len(cov) else np.nan})
    s = pd.DataFrame(rows)
    s["m_ord"] = s.model.map(lambda m: (ORDER + [NAIVE]).index(m))
    s["v_ord"] = s.variant.map(lambda v: VARIANTS.index(v) if v in VARIANTS else -1)
    return s.sort_values(["rule", "m_ord", "v_ord"]).drop(columns=["m_ord", "v_ord"]).reset_index(drop=True)


def paired_all(c: pd.DataFrame) -> pd.DataFrame:
    out = []
    for rule in RULES:
        for subset, cc in (("all", c), ("post-t0", c[c.post_t0]), ("H2-2026", c[c.target.isin(H2)])):
            dd = cc.assign(model=cc.model + " [" + cc.variant + "]")
            dd.loc[cc.model == NAIVE, "model"] = NAIVE
            for m in [x for x in ORDER if x in set(cc.model)]:
                pairs = [(f"{m} [baseline]", f"{m} [{PRIMARY}]"), (NAIVE, f"{m} [{PRIMARY}]"), (NAIVE, f"{m} [baseline]")]
                pairs += [(f"{m} [baseline]", f"{m} [{v}]") for v in VARIANTS if v not in ("baseline", PRIMARY)]
                for a, b in pairs:
                    p = forecast_eval.paired(dd, a, b, rule)
                    if p:
                        out.append({"subset": subset, **p})
    return pd.DataFrame(out)


def ranking(s: pd.DataFrame) -> pd.DataFrame:
    """Per model and rule, variants ordered by mean |log error| (all, post-t0)."""
    rows = []
    for (rule, model), g in s[s.variant.isin(VARIANTS)].groupby(["rule", "model"], sort=False):
        if g.variant.nunique() < 2:
            continue
        for col in ("mae_all", "mae_post"):
            o = g.sort_values(col)
            rows.append({"rule": rule, "model": model, "on": col.replace("mae_", ""),
                         "order (best first)": " < ".join(f"{v} {x:.3f}" for v, x in zip(o.variant, o[col])),
                         "best": o.variant.iloc[0], f"{PRIMARY} rank": int(list(o.variant).index(PRIMARY)) + 1
                         if PRIMARY in set(o.variant) else np.nan})
    return pd.DataFrame(rows)


def q4_table(path=config.REGISTRY_PATH) -> pd.DataFrame:
    rows = []
    for e in (json.loads(l) for l in open(path) if l.strip()):
        name = str(e.get("model", ""))
        for prefix in ("q4_frozen_", "q4_v2_", "q4_v2pub_"):
            if name.startswith(prefix) and not (prefix == "q4_v2_" and name.startswith("q4_v2pub_")):
                rp = e.get("runrate_points", {})
                rev = e.get("revenue", {}).get("2026-10-01..2026-12-31", {})
                rows.append({"set": prefix.rstrip("_"), "model": name[len(prefix):], "asof": e["asof"],
                             "rr_2026-11-30": rp.get("2026-11-30", {}).get("median"),
                             "rr_2026-12-31": rp.get("2026-12-31", {}).get("median"),
                             "rr_2026-12-31 q10": rp.get("2026-12-31", {}).get("q10"),
                             "rr_2026-12-31 q90": rp.get("2026-12-31", {}).get("q90"),
                             "rev_q4_mean": rev.get("mean"), "rev_q4 q10": rev.get("q10"), "rev_q4 q90": rev.get("q90")})
    return pd.DataFrame(rows)


def reproduction() -> pd.DataFrame:
    """Fresh baseline runs (2026-11-13) vs the reports of 2026-11-10: max relative difference of the forecast."""
    old = {"A": ("forecast_backtest_{r}_2026-11-10.csv", "forecast_claims_only"),
           "B": ("forecast_backtest_{r}_2026-11-10.csv", "forecast_with_indicators"),
           "E": ("forecast_openrouter_or_price_{r}_2026-11-10.csv", "forecast"),
           "G": ("forecast_openrouter_gh_share_{r}_2026-11-10.csv", "forecast"),
           "R": ("forecast_openrouter_ramp_only_{r}_2026-11-10.csv", "forecast")}
    rows = []
    for rule in RULES:
        for m, (f, col) in old.items():
            fn = os.path.join(REP, f"forecast_interval_{m}_baseline_{rule}_{ASOF}.csv")
            if not os.path.exists(fn):
                continue
            o, n = pd.read_csv(os.path.join(REP, f.format(r=rule))), pd.read_csv(fn)
            j = o[["target", col]].rename(columns={col: "old"}).merge(
                n[["target", "forecast"]].rename(columns={"forecast": "new"}), on="target", how="outer")
            rows.append({"rule": rule, "model": LABELS[m], "n_old": len(o), "n_new": len(n),
                         "max_rel_diff": float(np.nanmax(np.abs(j.old / j.new - 1)))})
        fn = os.path.join(REP, f"forecast_interval_bayes_baseline_{rule}_{ASOF}.csv")
        if os.path.exists(fn):
            o, n = pd.read_csv(os.path.join(REP, f"forecast_bayes_{rule}_2026-11-10.csv")), pd.read_csv(fn)
            j = o.merge(n, on=["target", "method"], how="outer", suffixes=("_o", "_n"))
            rows.append({"rule": rule, "model": LABELS["bayes"], "n_old": len(o), "n_new": len(n),
                         "max_rel_diff": float(np.nanmax(np.abs(j.forecast_mu_o / j.forecast_mu_n - 1)))})
        fn = os.path.join(REP, f"forecast_mlfixed_baseline_{rule}_{ASOF}.csv")
        if os.path.exists(fn):
            o, n = pd.read_csv(os.path.join(REP, f"forecast_mlfixed_{rule}_2026-11-10.csv")), pd.read_csv(fn)
            k = ["company", "target", "model", "pool"]
            j = o[k + ["forecast"]].merge(n[k + ["forecast"]], on=k, how="outer", suffixes=("_o", "_n"))
            rows.append({"rule": rule, "model": "ml_fixed (all rows)", "n_old": len(o), "n_new": len(n),
                         "max_rel_diff": float(np.nanmax(np.abs(j.forecast_o / j.forecast_n - 1)))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plot
def plot(c: pd.DataFrame, path: str):
    """Forecast vs actual over time, baseline (hollow) vs iv115 (filled), joined by a thin line.
    Style of plot_signals.plot_forecasts; colors from its validated slots, shapes double them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator
    from runrate_nowcast.plot import AQUA, INK, INK2, SURFACE
    from runrate_nowcast.plot_signals import GREEN, MAGENTA, YELLOW, _style

    models = [(LABELS["A"], "A  claims only", AQUA, "s"), (LABELS["E"], "E  OpenRouter tokens x price", MAGENTA, "^"),
              (LABELS["G"], "G  GitHub new-repo share", GREEN, "o"), (LABELS["R"], "R  Ramp business adoption", YELLOW, "D")]
    obs = data.load().set_index("obs_id")
    d = c[c.model.isin([m[0] for m in models]) & c.variant.isin(["baseline", PRIMARY])].copy()
    d["date"] = [obs.loc[t, "period_end"] if m == "revenue" else obs.loc[t, "ref_date"] for t, m in zip(d.target, d.metric)]
    d["days"] = [(obs.loc[t, "period_end"] - obs.loc[t, "period_start"]).days if m == "revenue" else 0
                 for t, m in zip(d.target, d.metric)]
    d = d[~((d.metric == "revenue") & (d.days > 100))]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    cols = {"publication": "Forecast the day before each number was published",
            "period_end": "Forecast on the date the number refers to"}
    rows = {"run_rate": ("Run-rate, $B", [5, 10, 20, 40, 80, 160, 320], (4, 350)),
            "revenue": ("Quarterly revenue, $B", [0.25, 0.5, 1, 2, 4, 8, 16, 32], (0.2, 40))}
    span = pd.Timedelta(days=26)
    offs = np.linspace(-1, 1, len(models)) * span / 2
    for j, rule in enumerate(RULES):
        for i, metric in enumerate(("run_rate", "revenue")):
            ax = axes[i, j]
            _style(ax)
            ax.set_yscale("log")
            g = d[(d.rule == rule) & (d.metric == metric)]
            top = rows[metric][2][1] * 0.97
            for (key, label, color, mk), off in zip(models, offs):
                h = g[g.model == key].pivot_table(index=["target", "date"], columns="variant", values="forecast").reset_index()
                if not len(h):
                    continue
                x = pd.to_datetime(h.date) + off
                b, n = h["baseline"].clip(upper=top), h[PRIMARY].clip(upper=top)
                ax.vlines(x, np.minimum(b, n), np.maximum(b, n), color=color, lw=1.3, alpha=0.6)
                ax.scatter(x, b, marker=mk, s=42, facecolor=SURFACE, edgecolor=color, linewidths=1.4, zorder=4)
                ax.scatter(x, n, marker=mk, s=46, color=color, edgecolors=SURFACE, linewidths=1.0, zorder=5)
            act = g.drop_duplicates("target")
            ax.scatter(pd.to_datetime(act.date), act.actual, marker="_", s=420, color=INK, linewidths=2.4, zorder=6)
            for r in act.itertuples():
                ax.annotate(f"{r.actual:g}", (pd.Timestamp(r.date), r.actual), xytext=(16, -3), textcoords="offset points",
                            fontsize=7.5, color=INK2, va="top")
            ticks = rows[metric][1]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_yticklabels([f"{t:g}" for t in ticks])
            ax.set_ylim(*rows[metric][2])
            mae = g.assign(a=g.log_err.abs()).groupby(["model", "variant"]).a.mean()
            txt = "   ".join(f"{lab.split('  ')[0]} {mae.get((k, 'baseline'), np.nan):.2f}->{mae.get((k, PRIMARY), np.nan):.2f}"
                             for k, lab, _, _ in models)
            ax.text(0.01, 0.97, "mean |log error| baseline->" + PRIMARY + ":  " + txt, transform=ax.transAxes,
                    fontsize=7.5, color=INK2, va="top")
            if i == 0:
                ax.set_title(cols[rule], loc="left", fontsize=11, color=INK, pad=8)
            if j == 0:
                ax.set_ylabel(rows[metric][0], color=INK2, fontsize=9)
    for ax in axes[1]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    handles = [Line2D([], [], marker="_", ls="", color=INK, ms=16, markeredgewidth=2.4, label="Actual")]
    handles += [Line2D([], [], marker=mk, ls="", color=c_, markeredgecolor=SURFACE, ms=8, label=lab) for _, lab, c_, mk in models]
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, ms=7, label="hollow: one-sided (baseline)"),
                Line2D([], [], marker="o", ls="", color=INK2, ms=7, label=f"filled: [X, 1.15X] ({PRIMARY})")]
    fig.suptitle("'Over $X' claims as one-sided (hollow) vs [X, 1.15X] (filled): forecasts without the S-1 vs actual values (log scale)",
                 x=0.01, y=0.99, ha="left", fontsize=12, color=INK)
    fig.legend(handles=handles, loc="upper left", ncol=7, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.005, 0.965))
    fig.text(0.01, 0.005, "Thin line joins the two forecasts of the same model and target. Forecasts above the panel are drawn at its top. "
             "FY2025 left out of the quarterly panel. Data: reports/eval_interval_long_" + ASOF + ".csv",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- report
def fmt(df, nd=3):
    return df.round(nd).to_markdown(index=False)


def main():
    d = load_long()
    flags, mism = post_t0_flags(d[d.model == LABELS["A"]])
    d["post_t0"] = [flags.get((r, t), False) for r, t in zip(d.rule, d.target)]
    d.to_csv(os.path.join(REP, f"eval_interval_long_{ASOF}.csv"), index=False)
    if "post_t0_ml" not in d:
        d["post_t0_ml"] = np.nan
    ml_post = d.dropna(subset=["post_t0_ml"])
    ml_flag_diff = ml_post[ml_post.post_t0_ml.astype(bool) != ml_post.post_t0][["rule", "model", "variant", "target"]]

    c = common(d)
    s = summary(c)
    s_own = summary(d)
    s_ss = summary(common(d[d.model.isin([LABELS[k] for k in ("A", "E", "G", "R")] + [NAIVE])]))
    p = paired_all(c)
    p_ss = paired_all(common(d[d.model.isin([LABELS[k] for k in ("A", "E", "G", "R")] + [NAIVE])]))
    rk = ranking(s)
    s.to_csv(os.path.join(REP, f"eval_interval_summary_{ASOF}.csv"), index=False)
    pd.concat([p.assign(table="common, all models"), p_ss.assign(table="common, A E G R")]).to_csv(
        os.path.join(REP, f"eval_interval_paired_{ASOF}.csv"), index=False)
    png = plot(common(d[d.model.isin([LABELS[k] for k in ("A", "E", "G", "R")] + [NAIVE])]),
               os.path.join(REP, f"forecast_vs_actual_interval_{ASOF}.png"))

    L = [f"# 'Over $X' claims as an interval (sub-task N): backtest ({ASOF})", "",
         "Pre-registered in DATA_PLAN.md (子任务 N, commit abce18a) before any full rerun. Primary rule **iv115**: a",
         "`kind = lower` claim X means X <= true value <= 1.15 X (two-sided truncated-normal update in model.py,",
         "`config.LOWER_UPPER_RATIO`). Variants reported regardless of outcome: baseline (one-sided, the original rule),",
         "point (a point claim at X), iv110, iv125. Inputs never include the filings; parameters refit at every vintage;",
         "crossing position profiled. Errors are log(actual / forecast); + = forecast too low.", "",
         "Disclosure (from DATA_PLAN): before registering the rule, the orchestrator saw baseline / point / point-at-1.05X",
         "for model A on rr_2026_05, rev_2026q3, rr_2026_09_s1a and rr_2026_10_s1a (publication rule).", ""]
    for rule in RULES:
        g = c[c.rule == rule]
        L.append(f"- {rule}: {g.target.nunique()} common targets ({', '.join(sorted(g.target.unique()))}); "
                 f"post-t0: {', '.join(sorted(g[g.post_t0].target.unique()))}.")
    runs = d[d.model != NAIVE].groupby("model").variant.apply(lambda v: ", ".join(x for x in VARIANTS if x in set(v)))
    L.append("- Variants run: " + "; ".join(f"{m}: {runs[m]}" for m in ORDER if m in runs) + ".")
    if mism:
        L.append(f"- post-t0 flag recomputed differs from forecast_mlfixed_2026-11-10 for {mism} (the CSV flag is used).")
    if len(ml_flag_diff):
        L.append(f"- ml_fixed reruns flag post-t0 differently on: {ml_flag_diff.to_dict('records')} (fixed flags used).")
    L += ["", "## Mean |log error| by model x variant x rule (common targets of every model and variant)", "",
          fmt(s[["rule", "model", "variant", "n_all", "mae_all", "bias_all", "n_post", "mae_post", "bias_post",
                 "n_h2", "mae_h2", "cov80", "cov80_post"]]), "",
          "## A, E, G, R and naive only (their common targets: all state-space targets)", "",
          fmt(s_ss[["rule", "model", "variant", "n_all", "mae_all", "bias_all", "n_post", "mae_post", "n_h2", "mae_h2",
                    "cov80", "cov50"]]), "",
          "## Each model on its own targets", "",
          fmt(s_own[["rule", "model", "variant", "n_all", "mae_all", "n_post", "mae_post", "cov80"]]), "",
          "## Variant ranking per model (pre-registered primary: " + PRIMARY + ")", "", fmt(rk), "",
          "## Paired differences in |log error| (`B vs A`: negative = B better; 90% bootstrap interval)", "",
          "Common targets of all models:", "",
          fmt(p[p.comparison.str.contains(rf"\[{PRIMARY}\] vs|\[baseline\] vs {NAIVE}", regex=True)]), "",
          "A, E, G, R on all their targets, every variant vs baseline:", "", fmt(p_ss), ""]
    a = d[d.model.isin([LABELS[k] for k in ("A", "E", "G", "R")]) & d.variant.isin(["baseline", PRIMARY, "point"])]
    for rule in RULES:
        t = a[a.rule == rule].pivot_table(index=["target", "post_t0"], columns=["model", "variant"], values="log_err")
        t.columns = [f"{m.split(' ')[0]} {v}" for m, v in t.columns]
        nv = d[(d.model == NAIVE) & (d.rule == rule)].set_index("target").log_err
        t["naive"] = [nv.get(x, np.nan) for x, _ in t.index]
        L += [f"Per-target log error, {rule}:", "", t.round(3).to_markdown(), ""]
    oa = openai_long()
    if len(oa):
        so = oa.groupby(["rule", "model", "variant"]).log_err.apply(lambda x: x.abs().mean()).unstack("variant").round(3)
        n = oa.groupby(["rule", "model", "variant"]).size().unstack("variant")
        so, n = (x[[v for v in VARIANTS if v in x.columns]] for x in (so, n))
        L += ["## OpenAI targets (ml_fixed second test set), mean |log error|", "", so.to_markdown(), "",
              "n targets:", "", n.to_markdown(), ""]
    rep = reproduction()
    if len(rep):
        L += ["## Baseline reruns vs the 2026-11-10 reports (max relative difference of the forecasts)", "",
              "E differs because indicators.csv changed after its 2026-11-10 run (commit 7606f58 dropped the truncated",
              "oldest OpenRouter week); every comparison here uses the fresh baseline, same code and data as the variants.", "",
              rep.to_markdown(index=False, floatfmt=".2e"), ""]
    q4 = q4_table()
    if len(q4):
        L += ["## Q4 2026 registrations: Nov 30 / Dec 31 run-rate medians and Q4 revenue means ($B)", "",
              fmt(q4, 2), ""]
    L += [f"Chart: `{os.path.basename(png)}`.", ""]
    with open(os.path.join(REP, f"eval_interval_{ASOF}.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
