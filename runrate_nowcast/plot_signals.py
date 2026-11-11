"""Charts for the real-time signals and the no-filing forecasts.

python -m runrate_nowcast plot-signals  -> reports/signals_<asof>.png, reports/forecast_errors_nofiling_<asof>.png

1. Signals: each real-time signal indexed to 100 in the week of 2026-01-05, on one shared log
   axis, against the official run-rate (company statements and S-1/A) indexed the same way.
   Small multiples, one signal per panel, so the signal color never has to carry identity.
2. Forecast errors without the filings: log(forecast / actual) per target, for the main
   variants, at both vintage rules. Marker shapes double the colors (CVD floor band).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from . import altdata, config, data, forecast_eval, pricing
from .plot import AQUA, BLUE, GRID, INK, INK2, SURFACE

YELLOW, MAGENTA = "#eda100", "#e87ba4"          # categorical slots 4 and 5 (validated with 1-3, all pairs)
BASE = pd.Timestamp("2026-01-08")
START, END = pd.Timestamp("2025-11-15"), pd.Timestamp("2026-11-08")
Q3 = (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-09-30"))


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, which="major")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def _index(s: pd.Series, base=BASE) -> pd.Series:
    """Series / its log-linear interpolated value at `base`, x 100."""
    s = s.dropna().sort_index()
    x = (s.index - base).days.to_numpy(float)
    b = math.exp(np.interp(0.0, x, np.log(s.to_numpy(float))))
    return 100 * s / b


def _weekly_sum(raw, name):
    s = raw[raw.indicator == name].set_index("date").value.sort_index()
    w = s.resample("W-SUN").agg(["sum", "count"])
    return w[w["count"] == 7]["sum"]


def signal_series() -> dict:
    raw = altdata.load_raw()
    raw["date"] = pd.to_datetime(raw["date"])
    out = {}
    orr = raw[raw.indicator == "openrouter_anthropic_tokens"].set_index("date").value.sort_index()
    out["OpenRouter: Anthropic tokens per week"] = orr
    px = pricing.price_index(list(orr.index), "github")
    out["OpenRouter tokens x price per token (spend proxy)"] = orr * px.to_numpy()
    out["npm @anthropic-ai/sdk downloads per week"] = _weekly_sum(raw, "npm_anthropic_sdk")
    out["OpenCode (npm opencode-ai) downloads per week"] = _weekly_sum(raw, "npm_opencode")
    g = pd.read_csv(os.path.join(config.PKG_DIR, "data", "github_claude_commits.csv"), parse_dates=["week_start"])
    g = g[(g["query"] == "Claude") & g.total_count.notna()]
    out["GitHub commits co-authored by Claude (one week in four)"] = g.set_index(g.week_start + pd.Timedelta(days=3)).total_count.sort_index()
    weeks = pd.date_range("2025-12-14", END, freq="W-SUN")
    out["Price per token, measured Claude Code model mix"] = pricing.price_index(list(weeks), "github")
    return {k: v[(v.index >= START) & (v.index <= END)] for k, v in out.items()}


def revenue_points() -> pd.DataFrame:
    df = data.load()
    ids = ["rr_2025_12", "rr_2026_02", "rr_2026_03", "rr_2026_04", "rr_2026_05", "rr_2026_07", "rr_2026_09_s1a", "rr_2026_10_s1a"]
    r = df[df.obs_id.isin(ids)][["obs_id", "ref_date", "value", "kind"]].sort_values("ref_date")
    return r


def plot_signals(path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator

    sig = signal_series()
    rev = revenue_points()
    pts = rev[rev.kind == "point"].set_index("ref_date").value
    rev_idx = _index(pts)
    lb = rev[rev.kind == "lower"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, s) in zip(axes.flat, sig.items()):
        _style(ax)
        ax.axvspan(*Q3, color=GRID, alpha=0.55, lw=0)
        is_price = name.startswith("Price per token")
        y = 100 * s if is_price else _index(s)
        if not is_price:
            ax.plot(rev_idx.index, rev_idx.values, color=INK, lw=1.4, zorder=3)
            ax.scatter(rev_idx.index, rev_idx.values, s=22, color=INK, zorder=4, edgecolors=SURFACE, linewidths=1.2)
            b = math.exp(np.interp(0.0, (pts.index - BASE).days.to_numpy(float), np.log(pts.to_numpy(float))))
            ax.scatter(lb.ref_date, 100 * lb.value / b, marker="^", s=30, facecolor=SURFACE, edgecolor=INK, lw=1.2, zorder=4)
        else:
            ax.axhline(100, color=INK2, lw=0.8, ls=":")
        style = dict(color=BLUE, lw=2, zorder=5)
        if name.startswith("GitHub"):
            ax.plot(y.index, y.values, marker="o", ms=4.5, **style)
        else:
            ax.plot(y.index, y.values, **style)
        ax.set_yscale("log")
        ax.set_title(name, loc="left", fontsize=9.5, color=INK, pad=6)
        last = y.dropna()
        ax.annotate(f"{last.iloc[-1]:.0f}", (last.index[-1], last.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    ticks = [25, 50, 100, 200, 400, 800, 1600, 3200]
    for ax in axes.flat:
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels([str(t) for t in ticks])
        ax.set_ylim(25, 4200)
        ax.set_xlim(START, END + pd.Timedelta(days=20))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0)
    import matplotlib.dates as mdates
    axes[1, 0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0, 0].text(Q3[0] + (Q3[1] - Q3[0]) / 2, 3800, "Q3 2026", ha="center", va="top", fontsize=8, color=INK2)
    handles = [Line2D([], [], color=INK, lw=1.4, marker="o", ms=4, label="Official run-rate (company statements, S-1/A)"),
               Line2D([], [], marker="^", ls="", markerfacecolor=SURFACE, color=INK, label="'Over $65B' (lower bound)"),
               Line2D([], [], color=BLUE, lw=2, label="Signal (named in each panel)")]
    fig.suptitle("Which real-time signals saw the Q3 2026 slowdown?  Index, week of 2026-01-05 = 100 (log scale)",
                 x=0.01, y=0.99, ha="left", fontsize=12, color=INK)
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.005, 0.955))
    fig.text(0.01, 0.01, "Revenue rose from 47 (May) to 71 (Sep 30) and 81 (Oct 31). Price per token: unit value of the Opus/Sonnet mix in public "
             "GitHub commits x list prices, flat before the first measured week (Dec 2025).", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


MODELS = [  # label in forecast_eval, legend label, color, marker
    ("A claims only", "A  claims only", AQUA, "s"),
    ("B + indicators", "B  + SDK downloads", BLUE, "o"),
    ("C price-adjusted (github, gamma 1)", "C  SDK downloads x price per token", YELLOW, "D"),
    ("E OpenRouter tokens x price", "E  OpenRouter tokens x price per token", MAGENTA, "^"),
]


def _target_label(o) -> str:
    if o.metric == "revenue":
        days = (o.period_end - o.period_start).days
        name = f"FY{o.period_end:%Y}" if days > 100 else f"Q{(o.period_end.month - 1) // 3 + 1} {o.period_end:%Y}"
        return f"{name}\nrevenue {o.value:g}"
    return f"{o.ref_date:%b %Y}\nrun-rate {o.value:g}"


def plot_errors(path: str, asof: str = "2026-11-10"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = forecast_eval.collect(asof)
    obs = data.load().set_index("obs_id")
    d = d[d.model.isin([m[0] for m in MODELS])].copy()
    d["date"] = [obs.loc[t, "period_end"] if m == "revenue" else obs.loc[t, "ref_date"] for t, m in zip(d.target, d.metric)]
    d["y"] = -d.log_err                                   # log(forecast / actual): + = too high
    order = (d.drop_duplicates("target").assign(rev=lambda x: x.metric == "revenue")
             .sort_values(["date", "rev"]).target.tolist())
    xpos = {t: i for i, t in enumerate(order)}
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    titles = {"publication": "Forecast made the day before each number was published",
              "period_end": "Forecast made on the date the number refers to (a true forecast)"}
    pct = [-40, -20, 0, 25, 50, 100, 150]
    h2 = [xpos[t] for t in ("rev_2026q3", "rr_2026_09_s1a", "rr_2026_10_s1a") if t in xpos]
    for ax, rule in zip(axes, ("publication", "period_end")):
        _style(ax)
        g = d[d.rule == rule]
        if h2:
            ax.axvspan(min(h2) - 0.5, max(h2) + 0.5, color=GRID, alpha=0.45, lw=0)
        ax.axhspan(math.log(0.9), math.log(1.1), color=GRID, alpha=0.9, lw=0)
        ax.axhline(0, color=INK, lw=1)
        offs = np.linspace(-0.27, 0.27, len(MODELS))
        for (key, label, color, mk), off in zip(MODELS, offs):
            h = g[g.model == key]
            x = h.target.map(xpos).to_numpy(float) + off
            y = h.y.clip(upper=math.log(2.6)).to_numpy()
            ax.vlines(x, 0, y, color=color, lw=1.2, alpha=0.7)
            ax.scatter(x, y, marker=mk, s=50, color=color, edgecolors=SURFACE, linewidths=1.2, zorder=4)
        ax.set_yticks([math.log(1 + p / 100) for p in pct])
        ax.set_yticklabels([f"{p:+d}%" if p else "0" for p in pct])
        ax.set_ylim(math.log(0.55), math.log(2.7))
        mae = g.assign(a=g.log_err.abs()).groupby("model").a.mean()
        n = g.groupby("model").size()
        ax.set_title(titles[rule], loc="left", fontsize=11, color=INK, pad=8)
        ax.set_title("mean |log error|  " + "   ".join(f"{lab.split('  ')[0]} {mae.get(k, np.nan):.3f}"
                                                        for k, lab, _, _ in MODELS) + f"   (n={int(n.max())})",
                     loc="right", fontsize=8.5, color=INK2, pad=8)
        if h2:
            ax.text(np.mean(h2), math.log(2.6), "H2 2026", ha="center", va="top", fontsize=8, color=INK2)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([_target_label(obs.loc[t]) for t in order], fontsize=7.5)
    axes[1].set_xlim(-0.6, len(order) - 0.4)
    handles = [Line2D([], [], marker=mk, ls="", color=c, markeredgecolor=SURFACE, ms=8, label=lab) for _, lab, c, mk in MODELS]
    fig.suptitle("Forecasts without the S-1: error for each target  (+ = forecast too high; band = within 10%)",
                 x=0.01, y=0.99, ha="left", fontsize=12, color=INK)
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.005, 0.965))
    fig.text(0.01, 0.005, "Inputs exclude the S-1, its amendments and third-party numbers that may derive from them; parameters are refit at "
             "each vintage. Errors above +160% are drawn at the top edge. Data: reports/forecast_eval_" + asof + ".csv",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def run(asof: str = "2026-11-10"):
    p1 = plot_signals(os.path.join(config.OUTPUT_DIR, f"signals_{asof}.png"))
    p2 = plot_errors(os.path.join(config.OUTPUT_DIR, f"forecast_errors_nofiling_{asof}.png"), asof)
    return p1, p2
