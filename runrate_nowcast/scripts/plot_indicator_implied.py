# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/plot_indicator_implied.py [evidence]
"""Sub-task P: each indicator's implied revenue, in real time, against the claims (baseline vs P1).

For every weekly (monthly for GitHub) reading of an indicator, published on day p:
  vintage   p - 1: the filter runs on the rows public then (crossing position profiled at the vintage),
            with the parameters refit at the latest backtest vintage on or before it (the run's
            params_*.json; catalog priors before the first refit), and stops at the reading's date t
  implied   the revenue the new reading implies through the relation known at the vintage:
            log I_t = a_i + c_f + beta_i (l + s_seg)  =>  l_implied = (log I_t - a_i - c_f) / beta_i - s_seg,
            shown as a run-rate on the company definition, exp(l_implied + b) ($B annualized).
            Baseline: a_i is the last filtered intercept (random walk); P1: it is extrapolated with the
            learned slope.
  model     exp(l + b) at t from the same filter (what the model believed before the reading)
Indicators (each in its own model): npm SDK downloads (B), OpenCode downloads (OC), OpenRouter tokens x
price (E), GitHub new-repo share (G).
Outputs: reports/indicator_implied_2026-11-13.png and reports/indicator_slope_2026-11-13/implied_<evidence>.csv
"""
import warnings; warnings.simplefilter("ignore")
import math, os, sys
import numpy as np
import pandas as pd
from runrate_nowcast import backtest, config, data, lower_rule, robust
from runrate_nowcast import indicator_slope as isl
from runrate_nowcast.model import B as IB, Model
from runrate_nowcast.plot import BLUE, GRID, INK, INK2, ORANGE, SURFACE
sys.path.insert(0, os.path.dirname(__file__))
from onestep_indicator_slope import OUTDIR, fits, point_claims  # noqa: E402

PANELS = [("B", "npm_anthropic_sdk", "SDK: npm @anthropic-ai/sdk downloads (model B)"),
          ("OC", "npm_opencode", "OpenCode npm downloads (model OC)"),
          ("E", "openrouter_anthropic_tokens", "OpenRouter Anthropic tokens x price (model E)"),
          ("G", "github_repo_share_claude", "GitHub new-repo share of Claude (model G)")]
VARS = [("baseline", "baseline (intercept random walk)", BLUE), ("P1", "P1 (intercept with learned slope)", ORANGE)]


def implied_series(model, name, variant, evidence):
    isl.apply_variant(variant)
    lower_rule.apply(isl.LOWER)
    config.USE_FILINGS = isl.EVIDENCE[evidence]
    isl.use_model(model)
    df = isl.model_frame(model)
    F = fits(model, variant, evidence)
    unf = backtest.unfitted_params()
    rows = []
    readings = df[(df.metric == "indicator") & (df.indicator == name)].sort_values("published")
    for r in readings.itertuples():
        v = pd.Timestamp(r.published) - pd.Timedelta(days=1)
        prior = [f for f in F if f[0] <= v]
        p = prior[-1][2] if prior else unf
        rho = robust.best_crossing_position(df, unf, v)
        ev = data.select(data.with_crossing_position(df, rho), vintage=v)
        if not ((ev.metric == "indicator") & (ev.indicator == name)).any():
            continue                                   # first reading: no relation known yet
        t = pd.Timestamp(r.ref_date)
        if t < pd.Timestamp(config.MODEL_START):
            continue
        res = Model(p).run(ev, end=max(t, v), smooth=False)
        m, _ = res.state(t, smoothed=False)
        lay, spec = res.layout, p["indicators"][name]
        beta = float(spec["beta"])
        f = spec.get("factor") or ""
        a = m[lay.ind(name)] + (m[lay.fac(f)] if f else 0.0)
        seg = spec.get("segment", "total")
        s = m[lay.seg(seg)] if seg != "total" else 0.0
        l_imp = (math.log(r.value) - a) / beta - s
        rows.append({"model": model, "indicator": name, "variant": variant, "evidence": evidence, "date": t,
                     "published": r.published, "implied": math.exp(l_imp + m[IB]), "model_rr": math.exp(m[0] + m[IB]),
                     "beta": beta, "slope": float(m[lay.slp(name)]) if lay.slopes else np.nan})
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def plot(d, claims, path, evidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (model, name, title) in zip(axes.flat, PANELS):
        for var, label, color in VARS:
            s = d[(d.indicator == name) & (d.variant == var)].sort_values("date")
            ax.plot(s.date, s.implied.clip(0.3, 400), color=color, lw=1.6, marker="o", ms=2.5)
        ax.scatter(claims.t_eff, claims.value, s=26, color=INK, zorder=5)
        ax.set_yscale("log")
        ticks = [0.5, 1, 2, 5, 10, 20, 50, 100, 200]
        ax.set_ylim(0.4, 400)
        ax.set_yticks(ticks); ax.set_yticklabels([f"${t:g}B" for t in ticks]); ax.minorticks_off()
        ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-11-15"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        _style(ax)
        tail = d[(d.indicator == name) & (d.date >= "2026-07-01")].groupby("variant").implied.median()
        txt = "median since Jul 2026: " + ", ".join(f"{v} ${tail.get(v, np.nan):.0f}B" for v, _, _ in VARS)
        ax.text(0.01, 0.97, txt, transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    for ax in axes[:, 0]:
        ax.set_ylabel("run-rate implied by the reading ($B, log)", color=INK2, fontsize=8.5)
    handles = [Line2D([], [], color=c, lw=1.6, marker="o", ms=3, label=f"Implied by each reading: {lb}") for _, lb, c in VARS]
    handles.append(Line2D([], [], color=INK, marker="o", ls="", ms=5, label="Point run-rate claims (official, leaks, third party, S-1/A)"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.022, "Real time: each reading is inverted through the indicator-revenue relation the filter knew the "
             "day before it was published (rows public then; parameters from the latest per-vintage refit). "
             "Claims: point run-rate claims at their effective date.", fontsize=7.5, color=INK2)
    fig.text(0.01, 0.006, f"Lower claims as X to 1.15X; evidence set '{evidence}' (S-1 numbers "
             f"{'never inputs' if evidence == 'nofil' else 'inputs once public'}). Values clipped to $0.3B-$400B.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(evidence="nofil"):
    parts = []
    for model, name, _ in PANELS:
        for var, _, _ in VARS:
            s = implied_series(model, name, var, evidence)
            print(model, name, var, len(s), flush=True)
            parts.append(s)
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(os.path.join(OUTDIR, f"implied_{evidence}.csv"), index=False)
    pc = point_claims(data.load_all(indicators=False))
    claims = pc[pc.metric == "run_rate"]
    suffix = "" if evidence == "nofil" else f"_{evidence}"
    plot(d, claims, os.path.join(config.OUTPUT_DIR, f"indicator_implied{suffix}_{isl.ASOF}.png"), evidence)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["nofil"]))
