"""Static chart of the smoothed path against every claim (needs matplotlib)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .model import Z80

# Reference categorical slots 1-3 (validated all-pairs, light mode) + neutral ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot(res, sel, asof, path, start="2025-07-01", horizon_days=75):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    asof = pd.Timestamp(asof)
    start = pd.Timestamp(start)
    stop = asof + pd.Timedelta(days=horizon_days)
    idx = (res.dates >= start) & (res.dates <= stop)
    dates = res.dates[idx]
    ms, Ps = res.ms[idx], res.Ps[idx]
    l_mu, l_sd = ms[:, 0], np.sqrt(Ps[:, 0, 0])
    off_mu = np.array([res.log_runrate(d, "t28x13")[0] for d in dates])

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    hist = dates <= asof
    ax.fill_between(dates, np.exp(l_mu - Z80 * l_sd), np.exp(l_mu + Z80 * l_sd), color=BLUE, alpha=0.15,
                    linewidth=0)
    ax.plot(dates[hist], np.exp(l_mu[hist]), color=BLUE, lw=2)
    ax.plot(dates[~hist], np.exp(l_mu[~hist]), color=BLUE, lw=2, ls=(0, (4, 3)))
    ax.plot(dates[hist], np.exp(off_mu[hist]), color=ORANGE, lw=2)
    ax.plot(dates[~hist], np.exp(off_mu[~hist]), color=ORANGE, lw=2, ls=(0, (4, 3)))

    # GAAP quarters: annualized average over the quarter
    gq = sel[(sel.metric == "revenue") & (sel.tier == "gaap")]
    for r in gq.itertuples():
        days = (r.period_end - r.period_start).days + 1
        if days > 100 or r.period_start < start:
            continue
        ann = r.value * 365 / days
        ax.plot([r.period_start, r.period_end], [ann, ann], color=AQUA, lw=3, solid_capstyle="round")
        ax.annotate(f"Q{r.period_end.quarter}'{r.period_end:%y} \\${r.value:.2f}B → \\${ann:.1f}B/yr",
                    (r.period_end, ann), textcoords="offset points",
                    xytext=(4, -6), ha="left", va="top", fontsize=7.5, color=INK2)

    # claims
    rr = sel[(sel.metric == "run_rate") & (sel.ref_date >= start)]
    for r in rr.itertuples():
        marker = {"point": "o", "lower": "^", "upper": "v"}[r.kind]
        filled = r.tier in ("official",)
        ax.scatter([r.ref_date], [r.value], marker=marker, s=46, zorder=5,
                   facecolor=INK if filled else SURFACE, edgecolor=INK, linewidths=1.4)
        # When a higher claim sits within 10 days, put this label on the left.
        near = rr[(abs((rr.ref_date - r.ref_date).dt.days) < 10) & (rr.value > r.value)]
        below = len(near) > 0
        ax.annotate(f"{r.value:g}", (r.ref_date, r.value), textcoords="offset points",
                    xytext=(-8, -1) if below else (6, 4), ha="right" if below else "left",
                    va="center" if below else "bottom", fontsize=7.5, color=INK2)

    ax.axvline(asof, color=INK2, lw=0.8, ls=":")
    ax.text(asof, 3.2, f" as of {asof:%Y-%m-%d}", fontsize=8, color=INK2, va="bottom")
    ax.set_yscale("log")
    ticks = [3, 5, 10, 20, 30, 50, 70, 100]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"${t}B" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(3, 130)
    ax.set_xlim(start, stop)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_title("Anthropic run-rate: smoothed state vs. every claim (log scale, $B annualized)",
                 loc="left", fontsize=12, color=INK, pad=12)
    handles = [
        Line2D([], [], color=BLUE, lw=2, label="GAAP-basis run-rate (instantaneous), 80% band"),
        Line2D([], [], color=ORANGE, lw=2, label="Official-basis run-rate (t28×13 + gap b)"),
        Line2D([], [], color=AQUA, lw=3, label="GAAP quarterly revenue, annualized"),
        Line2D([], [], marker="o", ls="", color=INK, label="Official claim"),
        Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, color=INK, label="Leak / third-party claim"),
        Line2D([], [], marker="^", ls="", markerfacecolor=SURFACE, color=INK, label="'Over $X' (lower bound)"),
        Line2D([], [], color=INK2, lw=1.5, ls=(0, (4, 3)), label="Forecast"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
