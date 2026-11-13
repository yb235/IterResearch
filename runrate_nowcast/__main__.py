"""CLI.

python -m runrate_nowcast nowcast [--asof D] [--plot]   # report -> reports/
python -m runrate_nowcast fit [--write]                  # MAP process noise -> params.json
python -m runrate_nowcast calibrate [--write]            # per-tier bias/noise vs GAAP path
python -m runrate_nowcast backtest [--refit]             # vintage backtest + coverage
python -m runrate_nowcast alerts --since D [--asof D]    # new claims scored against the prior forecast
python -m runrate_nowcast register [--asof D] [--note T] # time-stamped forecast -> forecasts/registry.jsonl
python -m runrate_nowcast score                          # score registered forecasts
python -m runrate_nowcast update-data                    # fetch npm / PyPI / Ramp indicators
python -m runrate_nowcast fit-indicators [--write]       # MAP beta / drift / noise per indicator
python -m runrate_nowcast layers [--refit]               # backtest one layer at a time
python -m runrate_nowcast audit                          # sensitivity of conclusions to hand-set choices
python -m runrate_nowcast ml                             # ML on interpolated labels: backtest + current nowcast
python -m runrate_nowcast bayes [--lags]                 # posterior (MAP + Laplace) nowcast
python -m runrate_nowcast bayes-backtest                 # strict backtest with posterior refit per vintage
python -m runrate_nowcast plot-errors                    # charts: actual vs forecast over time, forecast evolution
python -m runrate_nowcast fit-crossing [--write]         # where "crossed $X" milestones sit in their windows
python -m runrate_nowcast leakcheck [--sensitivity]       # can any forecast see what it should not?
python -m runrate_nowcast forecast-eval                   # no-filing forecast backtests, one table
python -m runrate_nowcast plot-signals                    # charts: real-time signals vs revenue; no-filing forecast errors
python -m runrate_nowcast price-study run --variant C --gamma 1 --rule publication   # one component backtest
python -m runrate_nowcast price-study combine             # gamma by evidence, variants A-D, paired bootstrap
python -m runrate_nowcast price-study nowcast --asof D [--variant C] [--register]   # Q4 2026 and 12/31 run-rate
"""
from __future__ import annotations

import argparse
import os
import warnings

import pandas as pd

from . import backtest, calibrate, config, data, registry, report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="runrate_nowcast")
    ap.add_argument("--data", default=config.DATA_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("nowcast"); p.add_argument("--asof", default=None); p.add_argument("--plot", action="store_true")
    p.add_argument("--no-robust", action="store_true", help="skip the specification-averaged headline")
    p.add_argument("--z-sd", type=float, default=None, help="out-of-sample z sd from `layers --refit` (variant C)")
    p = sub.add_parser("fit"); p.add_argument("--write", action="store_true")
    p = sub.add_parser("calibrate"); p.add_argument("--write", action="store_true")
    p.add_argument("--min-points", type=int, default=3)
    p = sub.add_parser("backtest"); p.add_argument("--refit", action="store_true")
    p = sub.add_parser("alerts"); p.add_argument("--since", required=True); p.add_argument("--asof", default=None)
    p = sub.add_parser("register"); p.add_argument("--asof", default=None); p.add_argument("--note", default="")
    sub.add_parser("score")
    p = sub.add_parser("update-data"); p.add_argument("--start", default="2025-01-01")
    p = sub.add_parser("fit-indicators"); p.add_argument("--write", action="store_true")
    p = sub.add_parser("layers"); p.add_argument("--refit", action="store_true")
    sub.add_parser("audit")
    p = sub.add_parser("ml"); p.add_argument("--asof", default=None)
    p = sub.add_parser("bayes"); p.add_argument("--asof", default=None)
    p.add_argument("--lags", action="store_true", help="also mix over indicator lag and definition lag by evidence")
    p.add_argument("--register", action="store_true", help="append the summary to forecasts/registry.jsonl")
    p = sub.add_parser("bayes-backtest"); p.add_argument("--asof", default=None)
    p.add_argument("--no-crossing-fit", action="store_true", help="keep the neutral crossing position (0.5)")
    p = sub.add_parser("plot-errors"); p.add_argument("--asof", default=None)
    p = sub.add_parser("fit-crossing"); p.add_argument("--write", action="store_true")
    p = sub.add_parser("leakcheck"); p.add_argument("--asof", default=None); p.add_argument("--sensitivity", action="store_true")
    p = sub.add_parser("forecast-eval"); p.add_argument("--asof", default=None)
    p = sub.add_parser("plot-signals"); p.add_argument("--asof", default=None)
    p = sub.add_parser("price-study"); p.add_argument("action", choices=["run", "combine", "nowcast"])
    p.add_argument("--variant", default="C"); p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--rule", default="publication"); p.add_argument("--asof", default=None)
    p.add_argument("--register", action="store_true")
    p.add_argument("--lower", default=None, help="lower-claim rule of sub-task N (lower_rule.VARIANTS), e.g. iv115")
    a = ap.parse_args(argv)

    warnings.simplefilter("ignore")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    if a.cmd == "price-study":
        from . import price_study as ps
        _price_study(a, ps)
        return
    if a.cmd == "update-data":
        from . import altdata
        altdata.update(start=a.start)
        print(f"-> {config.INDICATORS_PATH}")
        return
    df = data.load_all(a.data)
    params = calibrate.load_params()
    asof = pd.Timestamp(getattr(a, "asof", None) or pd.Timestamp.today().normalize())

    if a.cmd == "nowcast":
        md = report.markdown(df, params, asof, robust=not a.no_robust, z_sd_oos=a.z_sd)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        out = os.path.join(config.OUTPUT_DIR, f"nowcast_{asof:%Y-%m-%d}.md")
        with open(out, "w") as f:
            f.write(md)
        print(md)
        print(f"-> {out}")
        if a.plot:
            from .plot import plot, plot_layers
            _, res, sel = report.run_nowcast(df, params, asof)
            png = plot(res, sel, asof, os.path.join(config.OUTPUT_DIR, f"nowcast_{asof:%Y-%m-%d}.png"))
            png2 = plot_layers(res, sel, asof, os.path.join(config.OUTPUT_DIR, f"layers_{asof:%Y-%m-%d}.png"))
            print(f"-> {png}\n-> {png2}")

    elif a.cmd == "fit":
        fit = calibrate.fit_process_noise(df, params)
        print(fit)
        if a.write:
            params.update({k: fit[k] for k in ("sigma_l", "sigma_g", "sigma_b", "growth_reversion")})
            calibrate.save_params(params)
            print(f"-> {config.PARAMS_PATH}")

    elif a.cmd == "calibrate":
        resid, table = calibrate.calibrate_sources(df, params)
        print(resid.round(3).to_string(index=False))
        print()
        print(table.round(3).to_string(index=False))
        ov = calibrate.tier_overrides(table, a.min_points)
        print("\nOverrides (tiers with enough points):", ov or "none")
        if a.write and ov:
            params.setdefault("tiers", {}).update(ov)
            calibrate.save_params(params)
            print(f"-> {config.PARAMS_PATH}")

    elif a.cmd == "backtest":
        bt = backtest.run_backtest(df, params, refit=a.refit)
        print(bt.round(3).to_string(index=False))
        print(backtest.summarize(bt))

    elif a.cmd == "alerts":
        al = report.alerts(df, params, a.since, asof)
        print(al.round(3).to_string(index=False) if len(al) else "no new claims")

    elif a.cmd == "register":
        e = registry.register(df, params, asof, a.note, data_path=a.data)
        print({k: e[k] for k in ("registered_at_utc", "asof", "git_commit", "data_sha256", "n_obs")})
        for k, v in e["revenue"].items():
            print(f"  revenue {k}: {v['mean']:.2f} [{v['q10']:.2f}, {v['q90']:.2f}]")
        print(f"-> {config.REGISTRY_PATH} (commit it now)")

    elif a.cmd == "fit-indicators":
        out, fitted = calibrate.fit_indicators(df, params)
        if a.write:
            calibrate.save_params(fitted)
            print(f"-> {config.PARAMS_PATH}")

    elif a.cmd == "layers":
        tab, paired = backtest.layer_ablation(df, params, refit=a.refit)
        print(tab.round(3).to_string(index=False))
        print()
        print(paired.round(3).to_string(index=False))

    elif a.cmd == "ml":
        from . import ml
        res = ml.run(df, params=params, verbose=False)
        strict_path = os.path.join(config.OUTPUT_DIR, f"ml_vs_statespace_{asof:%Y-%m-%d}.csv")
        ss = pd.read_csv(strict_path)[["target", "SS_claims", "SS_indicators"]] if os.path.exists(strict_path) else None
        now = [ml.nowcast(df, params=params, asof=asof, label_kind=k) for k in ("interp", "smoother")]
        md = ml.markdown(res, ss, now, asof)
        out = os.path.join(config.OUTPUT_DIR, f"ml_{asof:%Y-%m-%d}.md")
        with open(out, "w") as f:
            f.write(md)
        print(md)
        print(f"-> {out}")

    elif a.cmd == "bayes":
        from . import bayes
        kw = {"lags": (-7, 0, 7), "def_lags": (7, 14)} if a.lags else {}
        summ, weights, post = bayes.posterior_nowcast(df, params, asof, **kw)
        md = [f"# Posterior nowcast (as of {asof:%Y-%m-%d})\n",
              "MAP + Laplace posterior over all estimated parameters; predictions mix over posterior draws.\n",
              "| Quantity | Median / mean | 80% interval |", "|---|---|---|"]
        for k, v in summ.items():
            c = v.get("median", v.get("mean"))
            md.append(f"| {k} | {c:.1f} | [{v['q10']:.1f}, {v['q90']:.1f}] |")
        md += ["", "## Structure weights (Laplace evidence)", "", weights.round(3).to_markdown(index=False), "",
               "## Parameter posterior (best structure)", "", post.round(4).to_markdown(index=False), ""]
        q = asof.to_period("Q")
        summ_reg = dict(summ, period=f"{q.start_time:%Y-%m-%d}..{q.end_time:%Y-%m-%d}")
        if a.register:
            e = registry.register_summary("bayes_laplace" + ("_lagmix" if a.lags else ""), asof, summ_reg,
                                          note="posterior (MAP + Laplace) nowcast", data_path=a.data)
            print(f"registered: {e['registered_at_utc']}")
        out = os.path.join(config.OUTPUT_DIR, f"bayes_{asof:%Y-%m-%d}.md")
        with open(out, "w") as f:
            f.write("\n".join(md))
        print("\n".join(md)); print(f"-> {out}")

    elif a.cmd == "bayes-backtest":
        from . import bayes
        bt = bayes.bayes_backtest(df, params, crossing=None if a.no_crossing_fit else "profile")
        suffix = "_rho05" if a.no_crossing_fit else ""
        bt.to_csv(os.path.join(config.OUTPUT_DIR, f"bayes_backtest{suffix}_{asof:%Y-%m-%d}.csv"), index=False)
        print(bayes.summarize(bt).round(3).to_string())

    elif a.cmd == "fit-crossing":
        from . import robust
        g = robust.crossing_profile(df, params)
        print(g.round(3).to_string(index=False))
        rho = float((g.rho * g.weight).sum())
        print(f"posterior mean crossing position: {rho:.3f}")
        if a.write:
            params["crossing_position"] = round(rho, 3)
            calibrate.save_params(params)
            print(f"-> {config.PARAMS_PATH}")

    elif a.cmd == "plot-errors":
        from . import plot_backtest
        d, p1, p2, extra = plot_backtest.run(f"{asof:%Y-%m-%d}")
        for path in [p1, p2] + extra:
            print(f"-> {path}")

    elif a.cmd == "audit":
        from . import audit
        base, tab = audit.run_audit(df, params)
        md = audit.markdown(base, tab, asof)
        out = os.path.join(config.OUTPUT_DIR, f"audit_{asof:%Y-%m-%d}.md")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(out, "w") as f:
            f.write(md)
        print(f"-> {out}")

    elif a.cmd == "leakcheck":
        from . import leakcheck
        md = leakcheck.run(df, params, sensitivity=a.sensitivity)
        out = os.path.join(config.OUTPUT_DIR, f"leakcheck_{asof:%Y-%m-%d}.md")
        with open(out, "w") as f:
            f.write(md)
        print(md); print(f"-> {out}")

    elif a.cmd == "forecast-eval":
        from . import forecast_eval
        d = forecast_eval.collect(f"{asof:%Y-%m-%d}")
        d = pd.concat([d, forecast_eval.combine_realtime(d)], ignore_index=True)
        d.to_csv(os.path.join(config.OUTPUT_DIR, f"forecast_eval_{asof:%Y-%m-%d}.csv"), index=False)
        md = forecast_eval.markdown(d, f"{asof:%Y-%m-%d}")
        out = os.path.join(config.OUTPUT_DIR, f"forecast_eval_{asof:%Y-%m-%d}.md")
        with open(out, "w") as f:
            f.write(md)
        print(md); print(f"-> {out}")

    elif a.cmd == "plot-signals":
        from . import plot_signals
        for path in plot_signals.run(f"{asof:%Y-%m-%d}"):
            print(f"-> {path}")

    elif a.cmd == "score":
        sc = registry.score(df, params)
        print(sc.round(3).to_string(index=False) if len(sc) else "nothing to score yet")


def _price_study(a, ps):
    import json
    if a.action == "run":
        print("->", ps.run_component(a.variant, a.gamma, a.rule, lower=a.lower))
    elif a.action == "combine":
        t = ps._tag(a.lower)
        d, summ, pairs = ps.combine(lower=a.lower)
        d.to_csv(os.path.join(config.OUTPUT_DIR, f"price_study{t}_{ps.ASOF}.csv"), index=False)
        summ.to_csv(os.path.join(config.OUTPUT_DIR, f"price_study_summary{t}_{ps.ASOF}.csv"), index=False)
        pairs.to_csv(os.path.join(config.OUTPUT_DIR, f"price_study_pairs{t}_{ps.ASOF}.csv"), index=False)
        if not t:
            ps.price_table().to_csv(os.path.join(config.OUTPUT_DIR, f"price_index_{ps.ASOF}.csv"), date_format="%Y-%m-%d")
            ps.decomposition().to_csv(os.path.join(config.OUTPUT_DIR, f"price_decomposition_{ps.ASOF}.csv"), index=False)
            ps.pricing.model_counts().to_csv(os.path.join(config.OUTPUT_DIR, f"github_model_counts_{ps.ASOF}.csv"))
        md = ps.markdown(d, summ, pairs, lower=a.lower)
        out = os.path.join(config.OUTPUT_DIR, f"price_study{t}_{ps.ASOF}.md")
        with open(out, "w") as f:
            f.write(md)
        print(md); print(f"-> {out}")
    elif a.action == "nowcast":
        asof = a.asof or f"{pd.Timestamp.today():%Y-%m-%d}"
        s = ps.nowcast(asof, a.variant, lower=a.lower)
        print(json.dumps({k: v for k, v in s.items() if k != "components"}, indent=1))
        for c in s["components"]:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
        out = os.path.join(config.OUTPUT_DIR, f"price_study_nowcast_{a.variant}{ps._tag(a.lower)}_{asof}.json")
        with open(out, "w") as f:
            json.dump(s, f, indent=1, default=str)
        print(f"-> {out}")
        if a.register:
            summ = {k: v for k, v in s.items() if k.startswith("runrate_") or k in ("revenue_quarter", "period")}
            w = ", ".join(f"{g:g}: {x:.2f}" for g, x in s["weights"].items())
            e = registry.register_summary(f"price_{a.variant}{ps._tag(a.lower)}", asof, summ,
                                          note=f"price study ({ps.LABELS.get(a.variant, a.variant)}); no filings as inputs; "
                                               f"lower claims: {a.lower or 'baseline (one-sided)'}; "
                                               f"gamma weights by claim likelihood {{{w}}}")
            print("registered", e["registered_at_utc"])


if __name__ == "__main__":
    main()
