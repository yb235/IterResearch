"""Forecast of one quarter's revenue as information arrives, v2 (sub-task P, item 1).

The previous chart (plot_backtest.evolution, reports/forecast_evolution_2026-11-10.png) deleted every
direct report of the plotted quarter's revenue, e.g. Bloomberg's 2026-08-17 leak of Q2 revenue $11.5B,
while its footnote said "using only data public on that date". This version draws both lines per model:

  solid   every row public on that date (the package's evidence rule: S-1 numbers are never inputs)
  dashed  the same rows minus direct reports of the plotted quarter's revenue (metric revenue, same period)

Models as in the previous chart: claims only (A frame) and claims + catalog indicators (B frame), the
current model with lower claims as [X, 1.15X] (config.LOWER_UPPER_RATIO = 1.15), and the full-sample
parameters of params.json (the only input not as of the date). Vintages: the weekly grid of the previous
chart plus every day a run-rate or revenue claim was published, so each jump sits on its news day.
"""
from __future__ import annotations

import pandas as pd

from . import calibrate, config, data
from .model import Model, period_key
from .plot import BLUE, GRID, INK, INK2, ORANGE, SURFACE

MODELS = [("claims", "Claims only", ORANGE), ("indicators", "Claims + SDK download indicators", BLUE)]
QUARTERS = [("2026Q2", ("2026-04-01", "2026-06-30"), "2026-03-08", "2026-10-25", 11.555, "2026-10-26"),
            ("2026Q3", ("2026-07-01", "2026-09-30"), "2026-06-07", "2026-11-01", 16.96, "2026-11-02")]


def frames():
    df_all = data.load_all()
    base = df_all[df_all.metric != "indicator"]
    return {"claims": base[(base.segment == "total") & (base.metric != "share")], "indicators": df_all}


def vintages(df, start, stop, freq="7D"):
    grid = set(pd.date_range(start, stop, freq=freq))
    sel = data.select(df, vintage=stop)
    news = sel[sel.metric.isin(["run_rate", "revenue"]) & (sel.published >= pd.Timestamp(start))]
    return sorted(grid | set(pd.to_datetime(news.published)))


def direct_mask(sel, period):
    ps, pe = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    return (sel.metric == "revenue") & (sel.period_start == ps) & (sel.period_end == pe)


def evolution_v2(frames_: dict, params: dict, period, vints) -> pd.DataFrame:
    """One row per vintage x model x line ("all" = every row public, "excl_direct" = minus direct reports)."""
    key = period_key(*period)
    end = pd.Timestamp(period[1])
    rows = []
    for v in vints:
        for label, df in frames_.items():
            sel = data.select(df, vintage=v)
            if (sel.published > v).any():
                raise AssertionError(f"row published after {v:%Y-%m-%d}")
            m = direct_mask(sel, period)
            out = {}
            for line, ev in (("all", sel), ("excl_direct", sel[~m])):
                if line == "excl_direct" and not m.any():
                    out[line] = out["all"]
                    continue
                r = Model(params).run(ev, end=max(v, end), queries=[period]).period_revenue(key)
                out[line] = r
            for line, r in out.items():
                rows.append({"vintage": v, "model": label, "line": line, "mean": r["mean"], "q10": r["q10"],
                             "q90": r["q90"], "n_direct": int(m.sum()), "direct": ";".join(sel[m].obs_id)})
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def plot(evos: list, path: str, raw: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(evos), figsize=(6.6 * len(evos), 5.6), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    notes = []
    for ax, (q, ev, actual, published, period) in zip(axes[0], evos):
        qend = pd.Timestamp(period[1])
        for key, label, color in MODELS:
            a = ev[(ev.model == key) & (ev.line == "all")].sort_values("vintage")
            x = ev[(ev.model == key) & (ev.line == "excl_direct")].sort_values("vintage")
            ax.fill_between(a.vintage, a.q10, a.q90, color=color, alpha=0.09, lw=0, step="post")
            ax.plot(a.vintage, a["mean"], color=color, lw=2, drawstyle="steps-post")
            ax.plot(x.vintage, x["mean"], color=color, lw=1.6, ls=(0, (4, 2.5)), drawstyle="steps-post")
        ax.axhline(actual, color=INK, lw=1.4)
        ax.text(ev.vintage.min(), actual, f" actual {actual:g} (S-1, published {pd.Timestamp(published):%b %d})",
                va="bottom", fontsize=8, color=INK)
        ax.axvline(qend, color=INK2, lw=0.8, ls=":")
        ylo, yhi = ax.get_ylim()
        ax.text(qend, yhi, " quarter ends", fontsize=7.5, color=INK2, va="top")
        # direct reports: mark the day they became public and the forecasts either side
        ids = sorted(set(";".join(ev.direct.dropna()).split(";")) - {""})
        for oid in ids:
            r = raw.loc[oid]
            d = pd.Timestamp(r.published)
            ax.axvline(d, color=INK2, lw=0.8)
            b = ev[(ev.model == "indicators") & (ev.vintage == d)].set_index("line")["mean"]
            txt = f"{d:%b %d}: report of {q[-2:]} revenue ${r.value:g}B\n+ indicators: {b['all']:.1f} with it, " \
                  f"{b['excl_direct']:.1f} without"
            ylo, yhi = ax.get_ylim()
            ax.annotate(txt, (d, b["all"]), xytext=(d + pd.Timedelta(days=6), actual + 0.42 * (yhi - actual)),
                        textcoords="data", fontsize=7.5, color=INK,
                        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.7})
        if not ids:
            notes.append(f"{q}: no direct report of the quarter's revenue was public before the S-1/A, so the "
                         f"solid and dashed lines coincide.")
        ax.set_title(f"{q[:4]} {q[4:]} revenue ($B): forecast as of each date", loc="left", fontsize=11, color=INK)
        ax.set_ylabel("$B per quarter; band = 80% interval (solid lines)", color=INK2, fontsize=8.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        _style(ax)
    handles = []
    for key, label, color in MODELS:
        handles.append(Line2D([], [], color=color, lw=2, label=f"{label}: rows usable under the rules, public on the date"))
        handles.append(Line2D([], [], color=color, lw=1.6, ls=(0, (4, 2.5)),
                              label=f"{label}: without direct reports of the quarter's revenue"))
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.0))
    foot = ["Solid: forecast from the rows public on that date that the rules allow. Public but never inputs: S-1 and "
            "S-1/A numbers, third-party figures that may come from them (Fortune, Sacra, SemiAnalysis), forward-looking "
            "statements (FT, NYT),",
            "and the 'already passed $100B' misquote. Dashed: the same rows minus direct reports of the plotted "
            "quarter's revenue. No new claim entered the evidence between Aug 17 and Nov 01, so the lines barely move there (weekly indicator readings only).",
            "'Over $X' claims as X to 1.15X. Ramp token data was first public on Nov 10, so it is never used here. "
            "Steps change on each weekly date and on each day a claim was published.",
            "Parameters: full-sample fit (params.json), not as of the date. In the strict backtest (parameters refit "
            "at each date) the indicator model's Q3 forecast on Nov 01 is 23.0, not the 20.1 drawn here."]
    foot += notes
    for i, t in enumerate(foot):
        fig.text(0.01, 0.006 + 0.022 * (len(foot) - 1 - i), t, fontsize=7.6, color=INK2)
    fig.tight_layout(rect=(0, 0.02 + 0.022 * len(foot), 1, 0.9))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def run(asof="2026-11-13", ratio=1.15):
    import os
    old = config.LOWER_UPPER_RATIO
    config.LOWER_UPPER_RATIO = ratio
    try:
        fr = frames()
        params = calibrate.load_params()
        raw = data.load().set_index("obs_id")
        evos, parts = [], []
        for q, period, start, stop, actual, published in QUARTERS:
            ev = evolution_v2(fr, params, period, vintages(fr["indicators"], start, stop))
            evos.append((q, ev, actual, published, period))
            parts.append(ev.assign(quarter=q))
        png = os.path.join(config.OUTPUT_DIR, f"forecast_evolution_v2_{asof}.png")
        plot(evos, png, raw)
        out = pd.concat(parts)
        out.to_csv(os.path.join(config.OUTPUT_DIR, f"forecast_evolution_v2_{asof}.csv"), index=False)
        return out, png
    finally:
        config.LOWER_UPPER_RATIO = old
