# Run from the repo root:
#   PYTHONPATH=. python runrate_nowcast/scripts/forecast_indicator_slope.py <model> <variant> <evidence> <rules>
#   e.g. ... forecast_indicator_slope.py B P1 nofil publication,period_end
"""Sub-task P backtests: models B, E, G, R, OC x variants (indicator_slope.VARIANTS) x evidence (nofil, pub).

Each run mirrors scripts/forecast_interval.py (no filings; models B, E, G, R), scripts/forecast_openrouter.py
(OC = its "opencode" variant) and scripts/forecast_pubfilings.py (evidence "pub": config.USE_FILINGS = True):
the same catalog patches and frames, the same call
    backtest.run_backtest(df, calibrate.load_params(), refit=True, crossing="profile", vintage_rule=rule)
with the lower-claim rule iv115 (config.LOWER_UPPER_RATIO = 1.15). The variant only sets the opt-in flags
config.INDICATOR_SLOPE / INDICATOR_DRIFT / INDICATOR_BETA_SD (indicator_slope.apply_variant).

Two recorders wrap package functions without changing what they return:
  backtest.prepublication_forecast  also keeps the parameters refit at the vintage (info["params"]) and the
                                    filtered indicator states at the vintage (online filter on the same evidence)
  backtest.check_no_leak            runs the package guard, then asserts on the raw observation table that
                                    every claim in the evidence was itself published on/before the vintage,
                                    and counts filing rows (filing != 0) in the evidence
Outputs, in reports/indicator_slope_2026-11-13/:
  bt_<model>_<variant>_<evidence>_<rule>.csv      run_backtest columns + variant/evidence + fitted parameters
  params_<model>_<variant>_<evidence>_<rule>.json the refit parameters per target (for real-time reuse)
Existing outputs are never overwritten.
"""
import warnings; warnings.simplefilter("ignore")
import json, math, os, sys, time
import numpy as np
import pandas as pd
from runrate_nowcast import backtest, calibrate, config, data, lower_rule
from runrate_nowcast import indicator_slope as isl
from runrate_nowcast.model import Model

OUTDIR = os.environ.get("ISL_OUTDIR") or os.path.join(config.OUTPUT_DIR, f"indicator_slope_{isl.ASOF}")
RAW = data.load().set_index("obs_id")
PARAMS_LOG: dict = {}
STATE_LOG: dict = {}
LEAK_LOG: dict = {}


def check_no_leak_logged(ev, vintage, row, _orig=backtest.check_no_leak):
    _orig(ev, vintage, row)                    # the package guard stays on
    v = pd.Timestamp(vintage)
    claims = ev[ev.metric != "indicator"]
    raw = RAW.loc[claims.obs_id]
    late = raw[raw.published > v]
    if len(late):
        raise AssertionError(f"{row.obs_id}: rows published after the vintage {v.date()}: {late.index.tolist()}")
    if (ev.published > v).any():
        raise AssertionError(f"{row.obs_id}: evidence published after the vintage")
    LEAK_LOG[row.obs_id] = int((raw.filing != 0).sum())


def prepublication_logged(df, row, *args, _orig=backtest.prepublication_forecast, **kw):
    info = kw.get("info")
    if info is None:
        info = {}
        kw["info"] = info
    out = _orig(df, row, *args, **kw)
    if out is None:
        return out
    mu, sd, n, ev = out
    p = info["params"]
    vint = pd.Timestamp(row.published) - pd.Timedelta(days=1)
    if kw.get("vintage") is not None:
        vint = min(pd.Timestamp(kw["vintage"]), vint)
    PARAMS_LOG[row.obs_id] = {"vintage": str(vint.date()), "params": p}
    res = Model(p).run(ev, end=vint, smooth=False)
    st = {}
    m, P = res.state(vint, smoothed=False)
    for name in res.layout.indicators:
        j = res.layout.ind(name)
        st[f"a_{name}"] = float(m[j])
        if res.layout.slopes:
            k = res.layout.slp(name)
            st[f"slope_{name}"], st[f"slope_sd_{name}"] = float(m[k]), float(math.sqrt(max(P[k, k], 0.0)))
    st["g_vintage"] = float(m[1])
    STATE_LOG[row.obs_id] = st
    return out


def main(model, variant, evidence, rules):
    for f in isl.FLAGS:
        assert getattr(config, f) is None, f"expected the package default config.{f} = None"
    assert not config.USE_FILINGS and config.LOWER_UPPER_RATIO is None
    isl.apply_variant(variant)
    lower_rule.apply(isl.LOWER)
    config.USE_FILINGS = isl.EVIDENCE[evidence]
    isl.use_model(model)
    backtest.check_no_leak = check_no_leak_logged
    backtest.prepublication_forecast = prepublication_logged
    df = isl.model_frame(model)
    p = calibrate.load_params()
    inds = df[df.metric == "indicator"].indicator.value_counts().to_dict()
    assert set(inds) == set(isl.INDICATORS[model]) or (model == "B" and set(inds) <= set(isl.INDICATORS["B"])), inds
    print(model, variant, evidence, lower_rule.describe(isl.LOWER),
          {f: getattr(config, f) for f in isl.FLAGS}, "USE_FILINGS", config.USE_FILINGS, inds, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    for rule in rules:
        stem = f"{model}_{variant}_{evidence}_{rule}"
        out = os.path.join(OUTDIR, f"bt_{stem}.csv")
        if os.path.exists(out):
            raise SystemExit(f"{out} exists; not overwriting")
        PARAMS_LOG.clear(); STATE_LOG.clear(); LEAK_LOG.clear()
        t = time.time()
        bt = backtest.run_backtest(df, p, refit=True, crossing="profile", vintage_rule=rule)
        bt.insert(0, "model", model)
        bt.insert(1, "variant", variant)
        bt.insert(2, "evidence", evidence)
        bt.insert(3, "rule", rule)
        for tg in bt.target:
            if PARAMS_LOG[tg]["vintage"] != str(bt[bt.target == tg].vintage.iloc[0]):
                raise AssertionError(f"{tg}: recorded vintage differs")
        bt["n_filing_evidence"] = [LEAK_LOG[t] for t in bt.target]
        for k in ("sigma_l", "sigma_g", "sigma_b", "growth_reversion"):
            bt[k] = [PARAMS_LOG[t]["params"].get(k, np.nan) for t in bt.target]
        for name in isl.INDICATORS[model]:
            for k in ("beta", "drift", "obs_sd"):
                bt[f"{k}_{name}"] = [PARAMS_LOG[t]["params"]["indicators"][name][k] for t in bt.target]
            bt[f"fitted_{name}"] = [bool(PARAMS_LOG[t]["params"]["indicators"][name].get("fitted", False)) for t in bt.target]
        extra = pd.DataFrame([STATE_LOG[t] for t in bt.target])
        bt = pd.concat([bt.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        bt.to_csv(out, index=False)
        with open(os.path.join(OUTDIR, f"params_{stem}.json"), "w") as f:
            json.dump(PARAMS_LOG, f, indent=1, sort_keys=True, default=float)
        print(stem, "done", len(bt), f"{time.time() - t:.0f}s", "mean log err", round(bt.log_err_model.mean(), 3),
              "share too high", round((bt.log_err_model < 0).mean(), 2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","))
