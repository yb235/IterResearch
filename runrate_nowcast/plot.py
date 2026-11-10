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


YELLOW = "#eda100"


def plot_layers(res, sel, asof, path, start="2025-07-01"):
    """Left: segment run-rates (official basis). Right: indicators vs model revenue, indexed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from . import config as cfg

    asof, start = pd.Timestamp(asof), pd.Timestamp(start)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    colors = {"api": BLUE, "subscription": ORANGE, "cloud": AQUA, "claude_code": YELLOW}
    days = pd.date_range(start, asof, freq="7D")

    # ---- segments
    for seg in res.layout.segments:
        mu = np.array([res.log_runrate(d, "t28x13", segment=seg)[0] for d in days])
        sd = np.array([res.log_runrate(d, "t28x13", segment=seg)[1] for d in days])
        ax1.fill_between(days, np.exp(mu - Z80 * sd), np.exp(mu + Z80 * sd), color=colors[seg], alpha=0.12, lw=0)
        ax1.plot(days, np.exp(mu), color=colors[seg], lw=2)
        ax1.annotate(cfg.SEGMENTS[seg]["label"].split(" (")[0], (days[-1], math.exp(mu[-1])), textcoords="offset points",
                     xytext=(5, 0), va="center", fontsize=8, color=INK)
    cc = sel[(sel.metric == "run_rate") & (sel.segment == "claude_code")]
    for r in cc.itertuples():
        ax1.scatter([r.ref_date], [r.value], marker="^" if r.kind == "lower" else "o", s=36, zorder=5,
                    facecolor=INK if r.tier == "official" else SURFACE, edgecolor=INK, linewidths=1.2)
    ax1.set_yscale("log")
    t1 = [0.5, 1, 2, 5, 10, 20, 50, 100]
    ax1.set_yticks(t1); ax1.set_yticklabels([f"${t:g}B" for t in t1]); ax1.minorticks_off()
    ax1.set_ylim(0.4, 130)
    ax1.set_xlim(start, asof + pd.Timedelta(days=75))
    ax1.set_title("Segment run-rates, official basis (80% bands)", loc="left", fontsize=11, color=INK)

    # ---- indicators vs model revenue, indexed to 100 at base date
    base = pd.Timestamp("2026-04-15")
    weeks = pd.date_range(start, asof, freq="7D")
    lrev = np.array([res.log_runrate(d, segment="api" if "api" in res.layout.segments else "total")[0] for d in weeks])
    lb = res.log_runrate(base, segment="api" if "api" in res.layout.segments else "total")[0]
    ax2.plot(weeks, 100 * np.exp(lrev - lb), color=INK, lw=2.4, label="Model: consumption revenue (GAAP basis)")
    icol = {"npm_anthropic_sdk": BLUE, "pypi_anthropic": ORANGE, "ramp_token_spend_anthropic": AQUA}
    specs = getattr(res, "indicator_specs", {}) or {}

    def res_breaks(name):
        return [pd.Timestamp(d) for d in specs.get(name, {}).get("breaks", [])]
    for name in res.layout.indicators:
        rows = sel[(sel.metric == "indicator") & (sel.indicator == name) & (sel.ref_date >= start)].sort_values("ref_date").copy()
        if not len(rows):
            continue
        label = name
        # Display only: chain-link across known breaks (the filter handles them via the intercept).
        for bdate in res_breaks(name):
            pre = rows[(rows.ref_date < bdate) & (rows.ref_date >= bdate - pd.Timedelta(days=14))]
            post = rows[(rows.ref_date >= bdate) & (rows.ref_date < bdate + pd.Timedelta(days=14))]
            if len(pre) and len(post):
                rows.loc[rows.ref_date < bdate, "value"] *= post.value.mean() / pre.value.mean()
                label = f"{name} (break-adjusted)"
        # index each indicator to the model revenue at its first reading on/after the base date
        ref = rows[rows.ref_date >= base].iloc[0] if (rows.ref_date >= base).any() else rows.iloc[0]
        lref = res.log_runrate(ref.ref_date, segment="api" if "api" in res.layout.segments else "total")[0]
        scale = 100 * math.exp(lref - lb) / ref.value
        if len(rows) > 2:
            ax2.plot(rows.ref_date, rows.value * scale, color=icol.get(name, INK2), lw=1.4, label=label)
        else:
            ax2.plot(rows.ref_date, rows.value * scale, color=icol.get(name, INK2), lw=1.4, marker="o", ms=6, label=label)
    ax2.set_yscale("log")
    t2 = [10, 20, 50, 100, 200, 400]
    ax2.set_yticks(t2); ax2.set_yticklabels([str(t) for t in t2]); ax2.minorticks_off()
    ax2.set_title(f"Leading indicators vs model revenue (index, {base:%b %Y} = 100)", loc="left", fontsize=11, color=INK)
    ax2.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.spines["bottom"].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        ax.axvline(asof, color=INK2, lw=0.8, ls=":")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
