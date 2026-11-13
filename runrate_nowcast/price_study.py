"""Does the price per token explain why the indicators overshoot? (2026-11-13)

Revenue = usage x price per token. The indicators (SDK downloads) measure usage; Anthropic cut
prices several times in 2026. If an indicator tracks usage, log(indicator) + gamma * log P tracks
revenue, with P the per-model price index of pricing.py (GitHub co-author trailer mix x list
price). gamma is not picked: gamma in {0, 0.5, 1} is averaged with weights prior x online-filter
likelihood of the claims public at each vintage (pricing.gamma_grid, backtest.gamma_mixture).

Variants (filings are never evidence; every parameter is refit at each vintage; crossing position
profiled per vintage):
  A       total-level claims only
  B       + raw indicators (npm / PyPI SDK downloads, Ramp token spend)       = C at gamma 0
  C       + price-adjusted indicators, gamma averaged by evidence (list prices)
  Ccache  C with the cache-heavy price basket (sensitivity)
  D       C + public GitHub commits with the Claude co-author trailer as one more usage
          indicator (drifting intercept, so it informs changes; wide prior beta 1 +/- 1)
Two vintages per target: the day before publication, and the date the number refers to.

Fixed before any C or D backtest was run (2026-11-13, ~11:20 UTC; A and B had already been run):
  - gamma grid {0, 0.5, 1} with a flat prior; weights from the claims' online-filter log
    predictive density at each vintage (ll_claims), with each component's parameters refit at
    that vintage. The joint likelihood (claims + indicators) is reported as a sensitivity only.
  - main price index: per-model mix x list price ("model"); the cache-heavy basket is a sensitivity.
  - main comparisons: C vs B, C vs A, D vs C, paired bootstrap of |log error|, both vintages.
  - the Q4 2026 / 2026-12-31 forecast uses the gamma-averaged variant among A, B, C, D with the
    lowest mean |log error| at the period-end vintage. This selection uses the evaluation targets,
    so the winner's backtest score is optimistic.
  - GitHub data: the per-model rows collected before DATA_CUTOFF (set when the price-tier queries
    finished, before running C or D).

python -m runrate_nowcast price-study run --variant C --gamma 1 --rule publication
python -m runrate_nowcast price-study combine
python -m runrate_nowcast price-study nowcast --asof 2026-11-13 [--register]
"""
from __future__ import annotations

import contextlib
import math
import os

import numpy as np
import pandas as pd

from . import altdata, backtest, calibrate, config, data, pricing

ASOF = "2026-11-10"              # label of the evaluation set (targets published by 2026-11-10)
# The GitHub backfill keeps running after the study; the study uses the per-model rows collected
# before this time (pricing.COLLECTED_BEFORE), so it can be rerun on exactly the same data.
DATA_CUTOFF = None
pricing.COLLECTED_BEFORE = DATA_CUTOFF
RULES = ("publication", "period_end")
COMPONENTS = [("A", 0.0), ("B", 0.0), ("C", 0.5), ("C", 1.0), ("Ccache", 0.5), ("Ccache", 1.0),
              ("D", 0.0), ("D", 0.5), ("D", 1.0)]
MIXTURES = {"C": {0.0: ("B", 0.0), 0.5: ("C", 0.5), 1.0: ("C", 1.0)},
            "Ccache": {0.0: ("B", 0.0), 0.5: ("Ccache", 0.5), 1.0: ("Ccache", 1.0)},
            "D": {0.0: ("D", 0.0), 0.5: ("D", 0.5), 1.0: ("D", 1.0)}}
SCHEME = {"A": "model", "B": "model", "C": "model", "Ccache": "model_cache", "D": "model"}
GITHUB_SPEC = {"role": "model", "beta": 1.0, "beta_sd": 1.0, "obs_sd": 0.15, "drift": 0.01,
               "aggregate": "none", "frequency": "weekly", "publish_delay_days": 1}
LABELS = {"A": "A claims only", "B": "B + raw indicators", "C": "C + price-adjusted indicators (gamma by evidence)",
          "Ccache": "C' cache-heavy price (sensitivity)", "D": "D = C + GitHub commits indicator"}


@contextlib.contextmanager
def github_indicator():
    """In this process only: GitHub commit totals enter the filter as a usage indicator."""
    orig = altdata.catalog

    def patched(path=config.CATALOG_PATH):
        c = orig(path).copy()
        m = c.indicator == "github_claude_commits"
        for k, v in GITHUB_SPEC.items():
            c.loc[m, k] = v
        return c
    altdata.catalog = patched
    try:
        yield patched
    finally:
        altdata.catalog = orig


def github_raw() -> pd.DataFrame:
    """Weekly totals of '"Co-Authored-By: Claude"' commits in the indicators.csv schema."""
    q = pricing.load_commit_queries()
    g = q[q["query"] == "Claude"]
    end = g.week_start + pd.Timedelta(days=6)
    return pd.DataFrame({"date": end, "indicator": "github_claude_commits", "value": g.total_count.astype(float),
                         "published": end + pd.Timedelta(days=1), "source": "github search/commits",
                         "note": "public commits in the week with a Claude co-author trailer (approximate)"})


def frame(variant: str) -> pd.DataFrame:
    """Observation table for a variant, before any price adjustment."""
    claims = data.load()
    if variant == "A":
        return claims[(claims.segment == "total") & (claims.metric != "share")]
    if variant in ("B", "C", "Ccache"):
        return data.load_all()
    if variant == "D":
        with github_indicator() as cat:
            raw = pd.concat([altdata.load_raw(), github_raw()], ignore_index=True)
            raw["date"] = pd.to_datetime(raw["date"])
            ind = altdata.indicator_observations(roles=("model",), cat=cat(), raw=raw)
        return pd.concat([claims, ind], ignore_index=True)
    raise ValueError(variant)


def component_path(variant, gamma, rule, asof=ASOF):
    return os.path.join(config.OUTPUT_DIR, f"price_study_{variant}_g{gamma:g}_{rule}_{asof}.csv")


def run_component(variant: str, gamma: float, rule: str, asof: str = ASOF) -> str:
    df = pricing.adjust_usage_indicators(frame(variant), gamma, SCHEME[variant])
    p = calibrate.load_params()
    ctx = github_indicator() if variant == "D" else contextlib.nullcontext()
    with ctx:
        if variant == "D":
            p["indicators"] = altdata.indicator_params()
        bt = backtest.run_backtest(df, p, refit=True, crossing="profile", vintage_rule=rule, loglik=True)
    bt["variant"], bt["gamma"], bt["rule"] = variant, gamma, rule
    path = component_path(variant, gamma, rule, asof)
    bt.to_csv(path, index=False)
    return path


def load_component(variant, gamma, rule, asof=ASOF) -> pd.DataFrame | None:
    path = component_path(variant, gamma, rule, asof)
    return pd.read_csv(path) if os.path.exists(path) else None


# --------------------------------------------------------------------------- combine
def combine(asof: str = ASOF):
    """Mixtures, fixed-gamma rows, and paired bootstraps. Returns (rows, summary, pairs)."""
    rows = []
    for rule in RULES:
        for v in ("A", "B"):
            c = load_component(v, 0.0, rule, asof)
            if c is not None:
                rows.append(c.assign(model=v, rule=rule))
        for v, spec in MIXTURES.items():
            comps = {g: load_component(*key, rule, asof) for g, key in spec.items()}
            if any(c is None for c in comps.values()):
                continue
            for lik in ("claims", "joint"):
                m = backtest.gamma_mixture(comps, likelihood=lik)
                rows.append(m.assign(model=v if lik == "claims" else f"{v}_jointlik", rule=rule))
            for g, c in comps.items():
                if g > 0:
                    rows.append(c.assign(model=f"{v}_g{g:g}", rule=rule))
    d = pd.concat(rows, ignore_index=True)
    keep = ["model", "rule", "target", "metric", "vintage", "actual", "forecast", "q10", "q90", "z", "pit",
            "in50", "in80", "log_err_model", "naive", "log_err_naive", "gamma_mean", "w_gamma_0", "w_gamma_0.5",
            "w_gamma_1"]
    d = d[[c for c in keep if c in d.columns]]
    summ = []
    for (m, rule), g in d.groupby(["model", "rule"]):
        summ.append({"model": m, "rule": rule, "n": len(g), "mae_log": g.log_err_model.abs().mean(),
                     "mean_err": g.log_err_model.mean(), "coverage80": g.in80.mean(), "coverage50": g.in50.mean(),
                     "sd_z": g.z.std(ddof=0), "gamma_mean": g.gamma_mean.mean() if "gamma_mean" in g else np.nan})
    summ = pd.DataFrame(summ)
    pairs = []
    for rule in RULES:
        e = {m: g.set_index("target").log_err_model for m, g in d[d.rule == rule].groupby("model")}
        for a, b in (("A", "B"), ("B", "C"), ("A", "C"), ("C", "D"), ("B", "D"), ("C", "Ccache"), ("B", "C_g1")):
            if a in e and b in e:
                pairs.append({"rule": rule, "comparison": f"{b} vs {a}", **backtest.paired_bootstrap(e[a], e[b])})
    return d, summ, pd.DataFrame(pairs)


def price_table(dates=None) -> pd.DataFrame:
    dates = pd.date_range("2025-11-03", "2026-11-16", freq="7D") if dates is None else dates
    return pd.DataFrame({"model_list": pricing.model_price_index(dates, "list"),
                         "model_cache": pricing.model_price_index(dates, "cache"),
                         "family_opus_sonnet": pricing.price_index(dates, "github")})


def _fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def markdown(d, summ, pairs, asof=ASOF) -> str:
    L = [f"# Price per token in the indicator layer: strict backtest ({asof} targets)", "",
         "Filings are never evidence; every parameter is refit at each vintage; crossing position profiled per vintage.",
         "Errors are log(actual / forecast): negative = forecast too high.", ""]
    order = ["A", "B", "C", "C_jointlik", "C_g0.5", "C_g1", "Ccache", "Ccache_g1", "D", "D_jointlik", "D_g1"]
    for rule in RULES:
        L += [f"## Vintage: {rule}", "", "| model | n | mean abs log err | mean err | 80% cover | 50% cover | sd z | mean gamma |",
              "|---|---|---|---|---|---|---|---|"]
        s = summ[summ.rule == rule].set_index("model")
        for m in order:
            if m in s.index:
                r = s.loc[m]
                L.append(f"| {m} | {r.n} | {_fmt(r.mae_log)} | {_fmt(r.mean_err)} | {_fmt(r.coverage80, 2)} | "
                         f"{_fmt(r.coverage50, 2)} | {_fmt(r.sd_z, 2)} | {_fmt(r.gamma_mean, 2)} |")
        L += ["", "| comparison | n | mean diff abs err | 90% CI | better / worse |", "|---|---|---|---|---|"]
        for r in pairs[pairs.rule == rule].itertuples():
            L.append(f"| {r.comparison} | {r.n} | {r.mean_diff_abs_err:+.3f} | [{r.ci90_lo:+.3f}, {r.ci90_hi:+.3f}] | "
                     f"{r.targets_improved} / {r.targets_worse} |")
        L += ["", "Per target (forecast, % error):", "", "| target | actual | " + " | ".join(
            m for m in ("A", "B", "C", "D") if m in s.index) + " | gamma weights (C: 0 / 0.5 / 1) |",
              "|---|---|" + "---|" * (len([m for m in ("A", "B", "C", "D") if m in s.index]) + 1)]
        sub = d[d.rule == rule]
        for t in sub[sub.model == "A"].sort_values("vintage").target:
            cells = []
            act = float(sub[sub.target == t].actual.iloc[0])
            for m in ("A", "B", "C", "D"):
                if m not in s.index:
                    continue
                x = sub[(sub.model == m) & (sub.target == t)]
                cells.append("" if not len(x) else f"{x.forecast.iloc[0]:.2f} ({100 * (x.forecast.iloc[0] / act - 1):+.0f}%)")
            c = sub[(sub.model == "C") & (sub.target == t)]
            w = "" if not len(c) else " / ".join(f"{c[f'w_gamma_{g}'].iloc[0]:.2f}" for g in ("0", "0.5", "1"))
            L.append(f"| {t} | {act:g} | " + " | ".join(cells) + f" | {w} |")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- decomposition
ANCHORS = [("rr_2025_12", "2025-12-31"), ("rr_2026_02", "2026-02-12"), ("rr_2026_04", "2026-04-03"),
           ("rr_2026_05", "2026-05-12"), ("rr_2026_09_s1a", "2026-09-30"), ("rr_2026_10_s1a", "2026-10-31")]
USAGE = ("npm_anthropic_sdk", "pypi_anthropic")


def _trailing(raw, name, date, days=28):
    x = raw[(raw.indicator == name) & (raw.date > pd.Timestamp(date) - pd.Timedelta(days=days))
            & (raw.date <= pd.Timestamp(date))].value
    return float(x.mean()) if len(x) else float("nan")


def decomposition(basket: str = "list") -> pd.DataFrame:
    """Between official run-rates: log growth of revenue vs usage indicators vs the price index.

    Evaluation only: the 2026-09-30 and 2026-10-31 run-rates come from the S-1/A. If revenue =
    usage x price, d log(revenue) = d log(usage) + d log(P), so the 'gap' d log(indicator) -
    d log(revenue) should equal -d log(P) when the indicator tracks usage one for one. Indicators
    are trailing 28-day means (the run-rate definition's window); PyPI is divided by its own step
    at the 2026-08-25 counting break (14-day means after vs before).
    """
    obs = data.load().set_index("obs_id")
    raw = altdata.load_raw()
    raw = raw[raw.value > 0].copy()
    b = pd.Timestamp("2026-08-25")
    py = raw[raw.indicator == "pypi_anthropic"].set_index("date").value
    step = py[(py.index >= b) & (py.index < b + pd.Timedelta(days=14))].mean() / \
        py[(py.index < b) & (py.index >= b - pd.Timedelta(days=14))].mean()
    m = (raw.indicator == "pypi_anthropic") & (raw.date >= b)
    raw.loc[m, "value"] = raw.loc[m, "value"] / step
    rows = []
    for (i0, d0), (i1, d1) in zip(ANCHORS[:-1], ANCHORS[1:]):
        r = {"from": d0, "to": d1, "dlog_revenue": math.log(obs.loc[i1, "value"] / obs.loc[i0, "value"])}
        for name in USAGE:
            r[f"dlog_{name}"] = math.log(_trailing(raw, name, d1) / _trailing(raw, name, d0))
        idx = pricing.model_price_index([pd.Timestamp(d0), pd.Timestamp(d1)], basket)
        r["dlog_price"] = math.log(idx.iloc[1] / idx.iloc[0])
        for name in USAGE:
            r[f"gap_{name}"] = r[f"dlog_{name}"] - r["dlog_revenue"]
        rows.append(r)
    out = pd.DataFrame(rows)
    tot = {"from": ANCHORS[0][1], "to": ANCHORS[-1][1]}
    for c in out.columns[2:]:
        tot[c] = out[c].sum()
    return pd.concat([out, pd.DataFrame([tot])], ignore_index=True)


# --------------------------------------------------------------------------- nowcast
def nowcast(asof: str, variant: str = "C", n_draws: int = 20000, seed: int = 0,
            dates=("2026-10-31", "2026-11-30", "2026-12-31"), period=("2026-10-01", "2026-12-31")):
    """Forecast with data public by `asof` (no filings), gamma averaged by evidence.

    For each gamma: crossing position profiled on data public by asof, every parameter refit
    on that data (as at a backtest vintage), smoothed run, then the claims' online-filter
    log-likelihood for the weight. The mixture is sampled like robust.averaged_nowcast.
    """
    from .model import Model, period_key
    from . import robust
    asof = pd.Timestamp(asof)
    spec = MIXTURES.get(variant, {0.0: (variant, 0.0)})
    comps = []
    for g, (v, _) in spec.items():
        base = frame(v)
        df = pricing.adjust_usage_indicators(base, g, SCHEME[v])
        ctx = github_indicator() if v == "D" else contextlib.nullcontext()
        with ctx:
            p = calibrate.load_params()
            p["indicators"] = altdata.indicator_params()
            rho = robust.best_crossing_position(df, backtest.unfitted_params(), asof)
            df = data.with_crossing_position(df, rho)
            vint = df[df.published <= asof]
            p.update({k: val for k, val in calibrate.fit_process_noise(vint, p).items()
                      if k in ("sigma_l", "sigma_g", "sigma_b", "growth_reversion")})
            if (vint.metric == "indicator").any():
                _, p = calibrate.fit_indicators(vint, p, verbose=False)
            sel = data.select(df, vintage=asof)
            ll = pricing.claim_loglik(Model(p).run(sel, smooth=False))
            res = Model(p).run(sel, end=max(pd.Timestamp(dates[-1]), pd.Timestamp(period[1])), queries=[period])
        c = {"gamma": g, "rho": rho, "ll_claims": ll, "n_obs": len(sel)}
        for d in dates:
            c[f"rr_{d}"] = res.log_runrate(d, "t28x13")
        q = res.period_revenue(period_key(*period))
        c["rev"] = (q["mean"], q["sd"])
        comps.append(c)
    w = pricing.gamma_weights_from_loglik({c["gamma"]: c["ll_claims"] for c in comps}).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(comps), size=n_draws, p=w)
    out = {"weights": dict(zip([c["gamma"] for c in comps], map(float, w)))}
    for d in dates:
        mu = np.array([comps[i][f"rr_{d}"][0] for i in idx])
        sd = np.array([comps[i][f"rr_{d}"][1] for i in idx])
        x = np.exp(rng.normal(mu, sd))
        out[f"runrate_{d}"] = {"median": float(np.median(x)), "q10": float(np.quantile(x, 0.1)),
                               "q90": float(np.quantile(x, 0.9))}
    m = np.array([comps[i]["rev"][0] for i in idx])
    s = np.array([comps[i]["rev"][1] for i in idx])
    s2 = np.log1p((s / m) ** 2)
    x = np.exp(rng.normal(np.log(m) - s2 / 2, np.sqrt(s2)))
    out["revenue_quarter"] = {"mean": float(x.mean()), "q10": float(np.quantile(x, 0.1)), "q90": float(np.quantile(x, 0.9))}
    out["period"] = period_key(*period)
    out["components"] = [{k: v for k, v in c.items()} for c in comps]
    return out
