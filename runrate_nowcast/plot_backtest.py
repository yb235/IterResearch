"""Charts of forecast errors: pre-publication forecasts vs actual values, and forecast evolution.

python -m runrate_nowcast plot-errors      -> reports/backtest_errors_<asof>.png, reports/forecast_evolution_<asof>.png
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from . import calibrate, config, data
from .model import Model, Z80, period_key
from .plot import AQUA, BLUE, GRID, INK, INK2, ORANGE, SURFACE

VIOLET = "#6250d6"
MODELS = [  # key, label, color, marker
    ("claims", "State-space, claims only", AQUA, "s"),
    ("indicators", "State-space + indicators", BLUE, "o"),
    ("bayes", "Bayesian, crossing position fitted", ORANGE, "^"),
    ("bayes_old", "Bayesian, crossing at window midpoint (previous)", ORANGE, "^"),
    ("ml", "ML ridge (no interval)", VIOLET, "D"),
]
HOLLOW = {"ml", "bayes_old"}


def _bayes_rows(path, actual):
    """Rebuild forecast and interval from the stored log error and z of the Bayesian backtest."""
    if not os.path.exists(path):
        return {}
    b = pd.read_csv(path)
    b = b[b.method == "laplace_mixture"]
    out = {}
    for r in b.itertuples():
        y = actual.get(r.target)
        if y is None:
            continue
        if "forecast_mu" in b.columns and pd.notna(r.forecast_mu):      # stored directly
            mu, sd = r.forecast_mu, r.forecast_sd
            if r.metric == "run_rate":
                out[r.target] = (math.exp(mu), math.exp(mu - Z80 * sd), math.exp(mu + Z80 * sd))
            else:
                out[r.target] = (mu, mu - Z80 * sd, mu + Z80 * sd)
            continue
        if r.metric == "run_rate":
            mu = math.log(y) - r.log_err
            sd = r.log_err / r.z if abs(r.z) > 1e-6 else np.nan
            out[r.target] = (math.exp(mu), math.exp(mu - Z80 * sd), math.exp(mu + Z80 * sd))
        else:
            mu = y / math.exp(r.log_err)
            sd = (y - mu) / r.z if abs(r.z) > 1e-6 else np.nan
            out[r.target] = (mu, mu - Z80 * sd, mu + Z80 * sd)
    return out


def collect(reports_dir=config.OUTPUT_DIR, asof="2026-11-10") -> pd.DataFrame:
    s = pd.read_csv(os.path.join(reports_dir, f"strict_backtest_{asof}.csv"))
    obs = data.load().set_index("obs_id")
    actual = dict(zip(s.target, s.actual))
    bayes = _bayes_rows(os.path.join(reports_dir, f"bayes_backtest_{asof}.csv"), actual)
    bayes_old = _bayes_rows(os.path.join(reports_dir, f"bayes_backtest_rho05_{asof}.csv"), actual)
    mlp = os.path.join(reports_dir, f"ml_vs_statespace_{asof}.csv")
    ml = pd.read_csv(mlp).set_index("target") if os.path.exists(mlp) else None
    rows = []
    for r in s.itertuples():
        o = obs.loc[r.target]
        date = o.period_end if r.metric == "revenue" else o.ref_date
        days = (o.period_end - o.period_start).days + 1 if r.metric == "revenue" else None
        base = {"target": r.target, "metric": r.metric, "date": pd.Timestamp(date), "vintage": pd.Timestamp(r.vintage),
                "actual": r.actual, "period_days": days}
        rows.append(dict(base, model="claims", forecast=r.forecast_claims_only, q10=r.q10_claims_only, q90=r.q90_claims_only))
        rows.append(dict(base, model="indicators", forecast=r.forecast_with_indicators,
                         q10=r.q10_with_indicators, q90=r.q90_with_indicators))
        for key, src in (("bayes", bayes), ("bayes_old", bayes_old)):
            if r.target in src:
                f, lo, hi = src[r.target]
                rows.append(dict(base, model=key, forecast=f, q10=lo, q90=hi))
        if ml is not None and r.target in ml.index:
            e = ml.loc[r.target, "ML_smoother_ridge"]
            f = math.exp(math.log(r.actual) - e) if r.metric == "run_rate" else r.actual / math.exp(e)
            rows.append(dict(base, model="ml", forecast=f, q10=np.nan, q90=np.nan))
    d = pd.DataFrame(rows)
    d["pct_err"] = d.forecast / d.actual - 1          # + = forecast above the actual
    return d


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def plot_errors(d: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), dpi=150, gridspec_kw={"height_ratios": [1.1, 1.1, 1]})
    fig.patch.set_facecolor(SURFACE)
    offsets = {m[0]: (i - 2) * 4 for i, m in enumerate(MODELS)}       # days, to keep markers apart

    for ax, metric, title, unit in (
            (axes[0], "run_rate", "Official run-rate claims: actual vs forecast made the day before publication", "$B annualized"),
            (axes[1], "revenue", "GAAP quarterly revenue: actual vs forecast made the day before the S-1", "$B per quarter")):
        sub = d[(d.metric == metric) & ((d.period_days.isna()) | (d.period_days < 100))]
        for key, label, color, marker in MODELS:
            m = sub[sub.model == key]
            x = m.date + pd.to_timedelta(offsets[key], unit="D")
            yerr = np.vstack([m.forecast - m.q10, m.q90 - m.forecast])
            ok = ~np.isnan(yerr).any(axis=0)
            if ok.any():
                ax.errorbar(x[ok], m.forecast[ok], yerr=yerr[:, ok], fmt="none", ecolor=color, elinewidth=1.4, alpha=0.75, capsize=0)
            hollow = key in HOLLOW
            ax.scatter(x, m.forecast, marker=marker, s=40, zorder=4, facecolor=SURFACE if hollow else color,
                       edgecolor=color, linewidths=1.4)
        act = sub.drop_duplicates("target")
        ax.scatter(act.date, act.actual, marker="_", s=420, color=INK, linewidths=2.4, zorder=5)
        for r in act.itertuples():
            ax.annotate(f"{r.actual:g}", (r.date, r.actual), textcoords="offset points", xytext=(12, -3),
                        fontsize=7.5, color=INK2, va="center")
        ax.set_yscale("log")
        ticks = [5, 7, 10, 15, 20, 30, 50, 70, 100] if metric == "run_rate" else [0.3, 0.5, 1, 2, 5, 10, 20]
        lo, hi = ax.get_ylim()
        ticks = [t for t in ticks if lo <= t <= hi]
        ax.set_yticks(ticks); ax.set_yticklabels([f"${t:g}B" for t in ticks]); ax.minorticks_off()
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylabel(unit, color=INK2, fontsize=9)
        _style(ax)

    # errors over time
    ax = axes[2]
    # Markers only: run-rate and revenue targets are different series, so no connecting lines.
    for key, label, color, marker in MODELS:
        m = d[d.model == key].sort_values("date")
        x = m.date + pd.to_timedelta(offsets[key], unit="D")
        ax.vlines(x, 0, 100 * m.pct_err, color=color, lw=1, alpha=0.35)
        ax.scatter(x, 100 * m.pct_err, marker=marker, s=36, zorder=4,
                   facecolor=SURFACE if key in HOLLOW else color, edgecolor=color, linewidths=1.3)
    ax.axhline(0, color=INK, lw=1)
    ax.axhspan(-10, 10, color=GRID, alpha=0.45, lw=0, zorder=0)
    for tgt, dy in (("rr_2026_05", 14), ("rev_2026q3", 14), ("rr_2025_10", 10)):
        w = d[(d.model == "indicators") & (d.target == tgt)]
        if len(w):
            w = w.iloc[0]
            kind = "run-rate" if w.metric == "run_rate" else "revenue"
            ax.annotate(f"{tgt} ({kind}): forecast {w.forecast:.1f} vs actual {w.actual:g}",
                        (w.date, 100 * w.pct_err), textcoords="offset points", xytext=(-8, dy), ha="right",
                        fontsize=8, color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.set_title("Forecast error by target date (+ = forecast too high; shaded = within ±10%)", loc="left",
                 fontsize=11, color=INK)
    ax.set_ylabel("forecast / actual − 1 (%)", color=INK2, fontsize=9)
    _style(ax)

    handles = [Line2D([], [], marker=mk, ls="", color=c, markerfacecolor=SURFACE if k in HOLLOW else c, label=lb)
               for k, lb, c, mk in MODELS]
    handles.append(Line2D([], [], marker="_", ls="", color=INK, markersize=16, markeredgewidth=2.4, label="Actual"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.0))
    g = d.assign(a=d.pct_err.abs()).groupby("model").a
    mae, n = g.mean(), g.count()          # count skips targets a model could not forecast
    txt = "Mean |error|: " + ",  ".join(f"{lb.split(' (')[0]} {100 * mae.get(k, np.nan):.0f}% (n={n.get(k, 0)})"
                                        for k, lb, _, _ in MODELS)
    fig.text(0.01, 0.018, txt + ".", fontsize=8, color=INK2)
    fig.text(0.01, 0.004, "Strict out-of-sample: each forecast uses only data public the day before the target was "
             "published; parameters are refit on that data (claims/indicators/Bayesian).", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def evolution(df_claims, df_all, params, period, start, stop, freq="7D"):
    """Weekly forecasts of one quarter's revenue as information arrives (data strictly as of each date)."""
    rows = []
    key = period_key(*period)
    for v in pd.date_range(start, stop, freq=freq):
        for label, df in (("claims", df_claims), ("indicators", df_all)):
            sel = data.select(df, vintage=v)
            sel = sel[~((sel.metric == "revenue") & (sel.period_start == pd.Timestamp(period[0]))
                        & (sel.period_end == pd.Timestamp(period[1])))]
            res = Model(params).run(sel, end=max(v, pd.Timestamp(period[1])), queries=[period])
            r = res.period_revenue(key)
            rows.append({"vintage": v, "model": label, "mean": r["mean"], "q10": r["q10"], "q90": r["q90"]})
    return pd.DataFrame(rows)


def plot_evolution(evos: list, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(evos), figsize=(6.2 * len(evos), 5.2), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, ev, actual, published, qend) in zip(axes[0], evos):
        for key, label, color, _ in MODELS[:2]:
            m = ev[ev.model == key]
            ax.fill_between(m.vintage, m.q10, m.q90, color=color, alpha=0.15, lw=0)
            ax.plot(m.vintage, m["mean"], color=color, lw=2, label=label)
        ax.axhline(actual, color=INK, lw=1.6)
        ax.text(ev.vintage.min(), actual, f" actual {actual:g} (published {published:%b %d})", va="bottom",
                fontsize=8, color=INK)
        ax.axvline(qend, color=INK2, lw=0.8, ls=":")
        ax.text(qend, ax.get_ylim()[1], " quarter ends", fontsize=7.5, color=INK2, va="top")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylabel("$B, 80% band", color=INK2, fontsize=9)
        _style(ax)
        ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.text(0.01, 0.01, "Each point: forecast using only data public on that date. Model parameters are the "
             "full-sample fit (data are strictly as of each date).", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def run(asof="2026-11-10"):
    d = collect(asof=asof)
    p1 = plot_errors(d, os.path.join(config.OUTPUT_DIR, f"backtest_errors_{asof}.png"))
    df_all = data.load_all()
    base = df_all[df_all.metric != "indicator"]
    df_claims = base[(base.segment == "total") & (base.metric != "share")]
    params = calibrate.load_params()
    ev3 = evolution(df_claims, df_all, params, ("2026-07-01", "2026-09-30"), "2026-06-07", "2026-11-01")
    ev2 = evolution(df_claims, df_all, params, ("2026-04-01", "2026-06-30"), "2026-03-08", "2026-10-25")
    p2 = plot_evolution([("Q2 2026 revenue: forecast as of each date", ev2, 11.555, pd.Timestamp("2026-10-26"),
                          pd.Timestamp("2026-06-30")),
                         ("Q3 2026 revenue: forecast as of each date", ev3, 16.96, pd.Timestamp("2026-11-02"),
                          pd.Timestamp("2026-09-30"))],
                        os.path.join(config.OUTPUT_DIR, f"forecast_evolution_{asof}.png"))
    d.to_csv(os.path.join(config.OUTPUT_DIR, f"backtest_errors_{asof}.csv"), index=False)
    pd.concat([ev2.assign(quarter="2026Q2"), ev3.assign(quarter="2026Q3")]).to_csv(
        os.path.join(config.OUTPUT_DIR, f"forecast_evolution_{asof}.csv"), index=False)
    return d, p1, p2
