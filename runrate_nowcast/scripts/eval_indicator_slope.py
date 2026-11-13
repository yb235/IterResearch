# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/eval_indicator_slope.py
"""Sub-task P: score the backtests of scripts/forecast_indicator_slope.py and onestep_indicator_slope.py.

Errors are log(actual / forecast): negative = forecast too high (backtest convention).
Per model x variant x vintage rule x evidence set, on the targets every variant of that model, rule and
evidence set forecast (they are the same targets by construction; checked):
  n, mean signed log error, share too high, mean |log error|, 80% and 50% coverage
  serial-correlation diagnostics of the errors in vintage order (ties: by the target's date): sign string
  ('-' = too high), number of same-sign runs, Wald-Wolfowitz runs-test p, lag-1 autocorrelation (acf1: the
  standard estimator used by Ljung-Box, overall mean and variance; acf1_pearson: corr(e[:-1], e[1:]))
  naive post-hoc correction: subtract from each forecast the mean log error of the targets whose actual was
  public at its vintage (indicator_slope.posthoc_corrected); its mean signed and mean |log error|
Paired comparisons (backtest.paired_bootstrap: 10000 draws, 90% interval, seed 0) of each variant against
the baseline on |log error|, and a paired bootstrap of the change in signed error (bias).
One-step-ahead (onestep_*.csv: every point claim as it arrives, publication order): n, mean z, share z < 0,
mean log error, lag-1 autocorrelation of z, Ljung-Box Q and p over lags 1-3, sign string; also by period.
The filter's own innovations of the point claims (innov_*.csv, effective-date order) get the same statistics.
Outputs (reports/): indicator_slope_summary_2026-11-13.csv, indicator_slope_paired_2026-11-13.csv,
indicator_slope_onestep_2026-11-13.csv (+ _onestep_long_), indicator_slope_innov_2026-11-13.csv,
indicator_slope_long_2026-11-13.csv, indicator_slope_params_2026-11-13.csv,
indicator_slope_onestep_2026-11-13.png; markdown tables printed to stdout (pasted into the report).
"""
import warnings; warnings.simplefilter("ignore")
import glob, os, re
import numpy as np
import pandas as pd
from runrate_nowcast import backtest, config, data
from runrate_nowcast import indicator_slope as isl

ASOF = isl.ASOF
D = os.path.join(config.OUTPUT_DIR, f"indicator_slope_{ASOF}")
REP = config.OUTPUT_DIR
RULES = ("publication", "period_end")
EVID = ("nofil", "pub")
VORDER = list(isl.VARIANTS)
MORDER = list(isl.MODELS)


def load_bt():
    parts = []
    for f in sorted(glob.glob(os.path.join(D, "bt_*.csv"))):
        parts.append(pd.read_csv(f))
    bt = pd.concat(parts, ignore_index=True)
    obs = data.load().set_index("obs_id")
    full = data.select(data.load(), filings=True).set_index("obs_id")
    bt["target_date"] = [obs.loc[t, "period_end"] if m == "revenue" else obs.loc[t, "ref_date"] for t, m in zip(bt.target, bt.metric)]
    bt["published"] = bt.target.map(full.published)
    bt["vintage"] = pd.to_datetime(bt.vintage)
    return bt


def check_targets(bt):
    for (m, r, e), g in bt.groupby(["model", "rule", "evidence"]):
        sets = {v: tuple(sorted(gg.target)) for v, gg in g.groupby("variant")}
        base = sets.get("baseline")
        for v, s in sets.items():
            if s != base:
                raise AssertionError(f"{m} {r} {e}: {v} has different targets from the baseline")


def summarize(g):
    g = g.sort_values(["vintage", "target_date", "target"])
    e = g.log_err_model.to_numpy()
    pub = dict(zip(g.target, g.published))
    pc = isl.posthoc_corrected(g.reset_index(drop=True), pub)
    return pd.Series({
        "n": len(g), "mean_err": e.mean(), "share_too_high": (e < 0).mean(), "mae": np.abs(e).mean(),
        "cov80": g.in80.mean(), "cov50": g.in50.mean(), "signs": isl.sign_string(e), "n_runs": isl.n_runs(e),
        "runs_p": isl.runs_test_p(e), "acf1": isl.acf(e, 1), "acf1_pearson": pd.Series(e).autocorr(1),
        "posthoc_mean_err": pc.mean(), "posthoc_mae": pc.abs().mean(),
        "n_corrected": int(sum(any((g.published <= v) & (g.target != t)) for v, t in zip(g.vintage, g.target))),
    })


def boot_signed(a, b, n_boot=10000, seed=0):
    common = a.index.intersection(b.index)
    d = (b[common] - a[common]).to_numpy()
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [5, 95])
    return d.mean(), lo, hi


def paired(bt):
    rows = []
    for (m, r, e), g in bt.groupby(["model", "rule", "evidence"]):
        base = g[g.variant == "baseline"].set_index("target").log_err_model
        for v in VORDER:
            if v == "baseline" or not (g.variant == v).any():
                continue
            x = g[g.variant == v].set_index("target").log_err_model
            pb = backtest.paired_bootstrap(base, x)
            s, slo, shi = boot_signed(base, x)
            rows.append({"model": m, "rule": r, "evidence": e, "comparison": f"{v} vs baseline", **pb,
                         "mean_diff_signed_err": s, "signed_ci90_lo": slo, "signed_ci90_hi": shi})
    return pd.DataFrame(rows)


def ljung(z):
    q, p = isl.ljung_box(z, 3)
    return q, p


def onestep():
    files = [f for f in sorted(glob.glob(os.path.join(D, "onestep_*_*_*.csv"))) if not f.endswith("_long.csv")]
    parts = [pd.read_csv(f) for f in files]
    if not parts:
        return None, None, None
    d = pd.concat(parts, ignore_index=True)
    d["published"] = pd.to_datetime(d.published)
    rows = []
    for (m, v, e), g in d.groupby(["model", "variant", "evidence"]):
        g = g.sort_values(["published", "t_eff", "obs_id"])
        z = g.z.to_numpy()
        q, p = ljung(z)
        early, late = g[g.published < "2026-05-01"], g[g.published >= "2026-05-01"]
        rows.append({"model": m, "variant": v, "evidence": e, "n": len(g), "mean_z": z.mean(), "share_z_neg": (z < 0).mean(),
                     "mean_log_err": g.log_err.mean(), "acf1_z": isl.acf(z, 1), "acf1_z_pearson": pd.Series(z).autocorr(1),
                     "ljung_box_q3": q, "ljung_box_p3": p,
                     "signs": isl.sign_string(z), "n_runs": isl.n_runs(z),
                     "mean_z_before_2026_05": early.z.mean(), "mean_z_from_2026_05": late.z.mean(),
                     "mean_log_err_from_2026_05": late.log_err.mean(), "n_from_2026_05": len(late)})
    inn = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "innov_*.csv")))]
    irows = []
    if inn:
        di = pd.concat(inn, ignore_index=True)
        for (m, v, e), g in di.groupby(["model", "variant", "evidence"]):
            z = g.sort_values("date").z.to_numpy()
            q, p = ljung(z)
            irows.append({"model": m, "variant": v, "evidence": e, "n": len(z), "mean_z": z.mean(),
                          "share_z_neg": (z < 0).mean(), "acf1_z": isl.acf(z, 1), "ljung_box_p3": p, "signs": isl.sign_string(z)})
    return d, pd.DataFrame(rows), pd.DataFrame(irows)


def params_table(bt):
    keep = ["model", "variant", "evidence", "rule", "target", "vintage"]
    cols = [c for c in bt.columns if re.match(r"^(beta|drift|obs_sd|slope|slope_sd|fitted)_", c)] + \
        ["sigma_l", "sigma_g", "sigma_b", "growth_reversion"]
    return bt[keep + cols].sort_values(["model", "variant", "evidence", "rule", "vintage"])


def plot_onestep(d, path, evidence="nofil"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from runrate_nowcast.plot import BLUE, GRID, INK, INK2, ORANGE, SURFACE
    VIOLET, AQUA = "#6250d6", "#1baf7a"
    cols = {"baseline": BLUE, "P1": ORANGE, "P1p": VIOLET, "P1pp": AQUA}
    fig, axes = plt.subplots(len(MORDER), 1, figsize=(12, 2.3 * len(MORDER) + 1), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    sub = d[d.evidence == evidence]
    for ax, m in zip(axes, MORDER):
        for i, (v, c) in enumerate(cols.items()):
            g = sub[(sub.model == m) & (sub.variant == v)].sort_values("published")
            x = pd.to_datetime(g.published) + pd.to_timedelta((i - 1.5) * 2.5, unit="D")
            ax.scatter(x, g.z.clip(-8, 8), s=16 if v in ("baseline", "P1") else 9, color=c, zorder=4,
                       alpha=1 if v in ("baseline", "P1") else 0.7)
        ax.axhline(0, color=INK, lw=0.9)
        ax.axhspan(-1.28, 1.28, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.set_ylim(-8.5, 8.5)
        ax.set_ylabel("z (clipped ±8)", color=INK2, fontsize=8)
        ax.set_title(isl.MODEL_LABELS[m], loc="left", fontsize=9.5, color=INK)
        ax.set_facecolor(SURFACE)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    handles = [Line2D([], [], marker="o", ls="", color=c, label=isl.LABELS[v]) for v, c in cols.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, "One-step-ahead standardized error of each point claim when it arrived: z = (actual - forecast) / sd, "
             "forecast from rows public the day before (negative = forecast too high). Shaded: |z| < 1.28 (80% band). "
             f"Evidence '{evidence}'.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def compact(s, e):
    """model x variant rows; per rule: mean signed error / share too high / mean |error| / 80% coverage."""
    rows = []
    t = s[s.evidence == e]
    for (m, v), g in t.groupby(["model", "variant"], observed=True):
        r = {"model": m, "variant": isl.LABELS[v]}
        for rule in RULES:
            x = g[g.rule == rule]
            if len(x):
                x = x.iloc[0]
                r[f"{rule} (n={int(x.n)})"] = f"{x.mean_err:+.3f} / {100 * x.share_too_high:.0f}% / {x.mae:.3f} / {x.cov80:.2f}"
                r[f"{rule} signs, runs, acf1"] = f"`{x.signs}` {int(x.n_runs)}, {x.acf1:+.2f}"
        rows.append(r)
    return pd.DataFrame(rows)


def md(df, floatfmt=".3f"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main():
    bt = load_bt()
    check_targets(bt)
    bt.to_csv(os.path.join(REP, f"indicator_slope_long_{ASOF}.csv"), index=False)
    s = bt.groupby(["model", "variant", "evidence", "rule"]).apply(summarize).reset_index()
    s["model"] = pd.Categorical(s.model, MORDER); s["variant"] = pd.Categorical(s.variant, VORDER)
    s = s.sort_values(["evidence", "model", "variant", "rule"])
    s.to_csv(os.path.join(REP, f"indicator_slope_summary_{ASOF}.csv"), index=False)
    pr = paired(bt)
    pr.to_csv(os.path.join(REP, f"indicator_slope_paired_{ASOF}.csv"), index=False)
    params_table(bt).to_csv(os.path.join(REP, f"indicator_slope_params_{ASOF}.csv"), index=False)
    d, os_sum, inn_sum = onestep()
    if d is not None:
        os_sum.to_csv(os.path.join(REP, f"indicator_slope_onestep_{ASOF}.csv"), index=False)
        d.to_csv(os.path.join(REP, f"indicator_slope_onestep_long_{ASOF}.csv"), index=False)
        if len(inn_sum):
            inn_sum.to_csv(os.path.join(REP, f"indicator_slope_innov_{ASOF}.csv"), index=False)
        plot_onestep(d, os.path.join(REP, f"indicator_slope_onestep_{ASOF}.png"))
    pd.set_option("display.width", 250)
    for e in EVID:
        t = s[s.evidence == e]
        if not len(t):
            continue
        print(f"\n### evidence {e}\n")
        print(md(t[["model", "variant", "rule", "n", "mean_err", "share_too_high", "mae", "cov80", "signs", "n_runs",
                    "acf1", "posthoc_mean_err", "posthoc_mae"]]))
    for e in EVID:
        print(f"\n### compact {e}: mean signed log error / share too high / mean |log error| / 80% coverage\n")
        print(compact(s, e).to_markdown(index=False))
    print("\n### paired\n")
    print(md(pr[["model", "rule", "evidence", "comparison", "n", "mean_diff_abs_err", "ci90_lo", "ci90_hi",
                 "targets_improved", "targets_worse", "mean_diff_signed_err", "signed_ci90_lo", "signed_ci90_hi"]]))
    if d is not None:
        print("\n### one-step\n")
        print(md(os_sum))
        if len(inn_sum):
            print("\n### filter innovations\n")
            print(md(inn_sum))


if __name__ == "__main__":
    main()
