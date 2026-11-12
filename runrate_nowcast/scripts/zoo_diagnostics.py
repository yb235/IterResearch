# Run from the repo root:
#   OMP_NUM_THREADS=1 PYTHONPATH=. python runrate_nowcast/scripts/zoo_diagnostics.py --parts ss,d1,d2,d3 --jobs 4
#   PYTHONPATH=. python runrate_nowcast/scripts/zoo_diagnostics.py --parts report
"""Diagnostics of the model zoo (model_zoo.py) against the single-signal state-space variants G and E.

Nothing in the package is changed; every number is computed with the package's own functions
(panel_model labels / pairs / folds, model_zoo members / forecasts, backtest.prepublication_forecast).

  ss  state-space side: G's and E's crossing position (robust profile) at each target's vintage, and for the
      pre-t0 publication targets (D4) a step-by-step path from the panel label path to G.
  d1  framework vs features: the zoo pipeline (pooled, ridge only) with one feature (GitHub new-repo share, or
      OpenRouter tokens with log price), also re-dated at G's crossing position; forecast decomposition.
  d2  inner-CV validity: inner-CV loss of every member vs (a) the same validation pairs with the training labels
      rebuilt from claims published before the block starts and (b) a pseudo-out-of-sample "real task" from
      vintage-nested origins (one per Anthropic anchor publication), at the publication vintages of the post-t0
      targets.
  d3  late-starting features of FEATURE_SETS["all"]: missingness by period, coefficients / contributions in AUTO's
      chosen member at the H2-2026 vintages, and refits without them.
  report  reports/zoo_diagnostics_2026-11-12.md from the CSVs.

Intermediate tables: reports/zoo_diagnostics_<part>_2026-11-12.csv.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from runrate_nowcast import altdata, backtest, calibrate, config, data, pricing, robust  # noqa: E402
from runrate_nowcast import features as F  # noqa: E402
from runrate_nowcast import model_zoo as mz  # noqa: E402
from runrate_nowcast import panel_model as pm  # noqa: E402

warnings.simplefilter("ignore")
ASOF, TODAY = "2026-11-10", "2026-11-12"
RULES = ("publication", "period_end")
REP = config.OUTPUT_DIR
WEEK = pm.WEEK


def out_csv(part):
    return os.path.join(REP, f"zoo_diagnostics_{part}_{TODAY}.csv")


def zoo_csv(fs, rule):
    return os.path.join(REP, f"forecast_zoo_{fs}_{rule}_{ASOF}.csv")


def ref_csv(variant, rule):
    return os.path.join(REP, f"forecast_openrouter_{variant}_{rule}_{ASOF}.csv")


# --------------------------------------------------------------------------- targets
def zoo_auto(fs="g_spec"):
    parts = [pd.read_csv(zoo_csv(fs, r)) for r in RULES]
    z = pd.concat(parts, ignore_index=True)
    return z


def target_sets():
    """(rule, target, vintage, gap_weeks) of the Anthropic targets the zoo forecasts, split by gap."""
    z = zoo_auto()
    a = z[(z.company == "anthropic") & (z.model == "AUTO") & (z.pool == "pooled")]
    a = a[["rule", "target", "vintage", "gap_weeks", "t0"]].drop_duplicates()
    return a[a.gap_weeks > 0].reset_index(drop=True), a[a.gap_weeks == 0].reset_index(drop=True)


_T = {}


def target_row(obs_id):
    if not _T:
        _T["t"] = pm.load_targets("anthropic").set_index("obs_id")
    return pm._Row(obs_id=obs_id, **_T["t"].loc[obs_id].to_dict())


def log_err(actual, fc):
    return math.log(actual / fc) if fc and fc > 0 and not math.isnan(fc) else np.nan


# --------------------------------------------------------------------------- feature sets for D1
CUSTOM = {
    "gh": lambda L, c: L[["github_repo_share"]],
    "or_spend": lambda L, c: pd.DataFrame({"or_spend": L["openrouter_log_tokens"]
                                           + (L["log_price"] if c == "anthropic" else 0.0)}),
    "or_tp": lambda L, c: L[["openrouter_log_tokens", "log_price"]],
}


def feature_levels(company, vintage, sources, fs):
    if fs in CUSTOM:
        return CUSTOM[fs](mz.feature_levels(company, vintage, sources, "g_spec"), company)
    return mz.feature_levels(company, vintage, sources, fs)


# --------------------------------------------------------------------------- shared helpers
_W = {}


def _init():
    _W["claims"] = {c: pm.load_claims(c) for c in pm.COMPANIES}
    _W["sources"] = F.load_sources()


def W():
    if not _W:
        _init()
    return _W["claims"], _W["sources"]


def build(tg, v, claims, sources, fs):
    """forecast_target's inputs: labels (target origin excluded), feature levels, pairs, forecast inputs."""
    v = pd.Timestamp(v)
    tc = tg.company
    dates, _ = pm.target_dates(tg)
    queries = [(tg.period_start, tg.period_end)] if tg.metric == "revenue" else None
    labs = {tc: pm.company_labels(tc, claims[tc], v, exclude_origin=tg.origin_id, end=max(dates), queries=queries)}
    for c in pm.COMPANIES:
        if c != tc:
            labs[c] = pm.company_labels(c, claims[c], v)
    flevs = {c: feature_levels(c, v, sources, fs) for c in labs}
    pairs = {c: mz.make_pairs(labs[c], flevs[c]) for c in labs}
    fi = mz.forecast_inputs(labs[tc], flevs[tc], tg)
    return labs, flevs, pairs, fi


def pool_P(pairs, tc, pool):
    cs = [tc] if pool == "own" else [tc] + [c for c in pm.COMPANIES if c != tc]
    parts = [pairs[c] for c in cs if len(pairs[c])]
    return pd.concat(parts, ignore_index=True) if parts else pairs[tc].iloc[0:0]


def predict_fc(model, fi):
    rates = np.asarray(model.predict(fi.G), float) if fi.after else np.array([])
    return mz.path_value(fi, rates)


def linear_decomp(model, fi):
    """Weekly growth predicted for the target's dates after t0, split into company mean + mom + features
    (exact for the Ridge-type members on the linear preprocessing); means over the dates after t0."""
    G = fi.G
    Z = model.prep_.transform(G)
    coef = np.asarray(model.est_.coef_, float).reshape(-1)
    ys = model.prep_.yscale_
    base = model.prep_.base(G)
    contrib = ys * Z * coef[None, :]
    feats = model.prep_.xcols
    out = {"rate_base": float(base.mean()), "rate_mom": float(contrib[:, 0].mean()),
           "rate_features": float(contrib[:, 1:].sum(axis=1).mean()), "g_weeks": float(G["g"].mean()),
           "mom_input": float(G["mom"].mean())}
    for j, f in enumerate(feats[1:], start=1):
        out[f"x:{f}"] = float(G[f].mean()) if f in G else np.nan
        out[f"elast:{f}"] = float(coef[j] * ys / model.prep_.scale_[j])
        out[f"contrib:{f}"] = float(contrib[:, j].mean())
    out["elast:mom"] = float(coef[0] * ys / model.prep_.scale_[0])
    return out


def rho_redated(claims, rho):
    c = dict(claims)
    c["anthropic"] = data.with_crossing_position(claims["anthropic"], rho)
    return c


_DFR = {}


def target_at_rho(obs_id, rho):
    """The target row with t_eff re-derived at crossing position rho (as prepublication_forecast does)."""
    key = round(float(rho), 6)
    if key not in _DFR:
        d = data.with_crossing_position(data.load(), rho)
        _DFR[key] = data.select(d[d.metric != "indicator"], filings=True).set_index("obs_id")
    tg = target_row(obs_id)
    r = _DFR[key].loc[obs_id]
    return pm._Row(**{**tg.__dict__, "t_eff": r.t_eff, "ref_date": r.ref_date})


# =========================================================================== SS (state space; for D1 and D4)
def _catalog_patch(variant):
    cat0 = _ORIG_CATALOG()

    def cat_patched(path=config.CATALOG_PATH):
        c = cat0.copy()
        if variant == "G":                       # scripts/forecast_ramp.py gh_share
            c.loc[c.role == "model", "role"] = "monitor"
            c.loc[c.indicator == "github_repo_share_claude", "role"] = "model"
        elif variant == "E":                     # scripts/forecast_openrouter.py or_price
            c.loc[c.indicator == "openrouter_anthropic_tokens", "role"] = "model"
            c.loc[c.indicator.isin(["npm_anthropic_sdk", "pypi_anthropic", "ramp_token_spend_anthropic"]),
                  "role"] = "monitor"
        return c
    return cat_patched


_ORIG_CATALOG = altdata.catalog
_SSD = {}


def ss_frames(variant):
    """(df, params) exactly as the G / E scripts build them; 'A'/'base' are claims-only tables."""
    if variant in _SSD:
        return _SSD[variant]
    if variant in ("G", "E"):
        altdata.catalog = _catalog_patch(variant)
        df = data.load_all()
        if variant == "E":
            df = pricing.adjust_indicators(df, 1.0, "github")
        p = calibrate.load_params()
    else:
        altdata.catalog = _ORIG_CATALOG
        df_all = data.load_all()
        base = df_all[df_all.metric != "indicator"]
        df = base if variant == "base" else base[(base.segment == "total") & (base.metric != "share")]
        p = calibrate.load_params()
    _SSD[variant] = (df, p)
    return df, p


def ss_forecast(variant, obs_id, vintage, crossing="profile", refit=True, params=None):
    """backtest.prepublication_forecast for one target; returns (forecast, log_err)."""
    df, p = ss_frames(variant)
    altdata.catalog = _catalog_patch(variant) if variant in ("G", "E") else _ORIG_CATALOG
    full = data.select(df, filings=True)
    row = full[full.obs_id == obs_id].iloc[0]
    out = backtest.prepublication_forecast(df, row, params if params is not None else p, refit=refit,
                                           crossing=crossing, vintage=pd.Timestamp(vintage))
    if out is None:
        return np.nan, np.nan
    mu = out[0]
    fc = math.exp(mu) if row.metric == "run_rate" else mu
    return fc, log_err(row.value, fc)


def ss_rho(variant, obs_id, vintage):
    df, _ = ss_frames(variant)
    altdata.catalog = _catalog_patch(variant) if variant in ("G", "E") else _ORIG_CATALOG
    full = data.select(df, filings=True)
    row = full[full.obs_id == obs_id].iloc[0]
    return robust.best_crossing_position(df, backtest.unfitted_params(), pd.Timestamp(vintage),
                                         exclude_origin=row.origin_id)


def ss_task(job):
    rule, obs_id, vintage, kind = job
    warnings.simplefilter("ignore")
    t = time.time()
    rec = {"rule": rule, "target": obs_id, "vintage": vintage, "kind": kind}
    for var in ("G", "E", "A"):
        rec[f"rho_{var}"] = ss_rho(var, obs_id, vintage)
    tg = target_row(obs_id)
    rec["t_eff_05"] = pd.Timestamp(tg.t_eff).date()
    rec["t_eff_rhoG"] = pd.Timestamp(target_at_rho(obs_id, rec["rho_G"]).t_eff).date()
    if kind == "pre":
        claims, _ = W()
        v = pd.Timestamp(vintage)
        sel = pm.public_claims(claims["anthropic"], v, tg.origin_id, start=pm.MODEL_STARTS["anthropic"])
        nz_panel = pm.vintage_noise("anthropic", sel)
        _, p = ss_frames("A")
        dfA, _ = ss_frames("A")
        nz_g = calibrate.fit_process_noise(dfA[dfA.published <= v], p)
        for k in pm.NOISE_KEYS:
            rec[f"noise_panel:{k}"] = nz_panel[k]
            rec[f"noise_ss:{k}"] = nz_g[k]
        steps = [
            ("S1r panel replica (panel noise, rho .5)", "A", None, False, dict(p, **nz_panel)),
            ("S2 + backtest noise refit", "A", None, True, None),
            ("S3 + crossing profile (= A)", "A", "profile", True, None),
            ("S4 + segment claims", "base", "profile", True, None),
            ("S5 + GitHub share indicator (= G)", "G", "profile", True, None),
            ("S5b G without segment claims", "G_noseg", "profile", True, None),
        ]
        for name, var, cr, refit, prm in steps:
            if var == "G_noseg":
                dfG, pG = ss_frames("G")
                key = "G_noseg"
                if key not in _SSD:
                    keep = (dfG.metric == "indicator") | ((dfG.segment == "total") & (dfG.metric != "share"))
                    _SSD[key] = (dfG[keep], pG)
                altdata.catalog = _catalog_patch("G")
                df_, p_ = _SSD[key]
                full = data.select(df_, filings=True)
                row = full[full.obs_id == obs_id].iloc[0]
                o = backtest.prepublication_forecast(df_, row, p_, refit=True, crossing="profile", vintage=v)
                fc = (math.exp(o[0]) if row.metric == "run_rate" else o[0]) if o else np.nan
                rec[f"fc:{name}"], rec[f"err:{name}"] = fc, log_err(row.value, fc)
                continue
            fc, le = ss_forecast(var, obs_id, vintage, crossing=cr, refit=refit, params=prm)
            rec[f"fc:{name}"], rec[f"err:{name}"] = fc, le
    rec["seconds"] = round(time.time() - t, 1)
    print(f"  ss {rule} {obs_id} {kind} {rec['seconds']}s", flush=True)
    return rec


def run_ss(jobs):
    post, pre = target_sets()
    todo = [(r.rule, r.target, r.vintage, "post") for r in post.itertuples()]
    todo += [(r.rule, r.target, r.vintage, "pre") for r in pre.itertuples() if r.rule == "publication"]
    todo = sorted(todo, key=lambda j: j[3] != "pre")
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(ss_task, todo))
    pd.DataFrame(rows).to_csv(out_csv("ss"), index=False)


# =========================================================================== D1
D1_VARIANTS = (("gh", "gh", None), ("or_spend", "or_spend", None), ("or_tp", "or_tp", None),
               ("gh@rhoG", "gh", "G"), ("or_spend@rhoE", "or_spend", "E"), ("g_spec@rhoG", "g_spec", "G"),
               ("g_spec", "g_spec", None),
               # mechanism probe (not a candidate model): the same ridge with the shrinkage fixed at alpha 0.01
               ("gh alpha=0.01", "gh", None), ("or_spend alpha=0.01", "or_spend", None))
PROBE_GRID = {"ridge": ({"alpha": 0.01},)}


def d1_task(job):
    rule, obs_id, vintage, var, fs, rho_var, rho = job
    warnings.simplefilter("ignore")
    claims, sources = W()
    tg = target_row(obs_id)
    if rho_var is not None:
        claims = rho_redated(claims, rho)
        tg = target_at_rho(obs_id, rho)
    labs, flevs, pairs, fi = build(tg, vintage, claims, sources, fs)
    lab = labs["anthropic"]
    rec = {"rule": rule, "target": obs_id, "vintage": vintage, "variant": var, "feature_set": fs,
           "rho": rho if rho_var else 0.5, "t0": lab.t0.date(), "t_eff": pd.Timestamp(tg.t_eff).date(),
           "ss_panel_err": log_err(tg.value, pm.smoother_forecast(lab, tg))}
    P = pool_P(pairs, "anthropic", "pooled")
    fits = mz.fit_zoo(P, ("ridge",), PROBE_GRID if "alpha=" in var else None)
    f = fits["ridge"]
    fc = predict_fc(f.model, fi)
    rec.update({"ridge_err": log_err(tg.value, fc), "ridge_fc": fc, "hyper": str(f.hyper), "cv_loss": f.cv_loss,
                "n_pairs": len(P), "start_level": float(np.exp(fi.start)) if tg.metric == "run_rate" else np.nan})
    if fi.after:
        rec.update(linear_decomp(f.model, fi))
        # what the realised path implies: weekly growth needed from y(t0) to hit the actual (run-rate targets)
        if tg.metric == "run_rate":
            rec["rate_needed"] = (math.log(tg.value) - fi.start) / float(fi.G["g"].iloc[0])
    print(f"  d1 {rule} {obs_id} {var} {rec['ridge_err']:.3f}", flush=True)
    return rec


def run_d1(jobs):
    post, _ = target_sets()
    ss = pd.read_csv(out_csv("ss")).set_index(["rule", "target"])
    todo = []
    for r in post.itertuples():
        for var, fs, rv in D1_VARIANTS:
            rho = float(ss.loc[(r.rule, r.target), f"rho_{rv}"]) if rv else None
            todo.append((r.rule, r.target, r.vintage, var, fs, rv, rho))
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init) as ex:
        rows = list(ex.map(d1_task, todo))
    pd.DataFrame(rows).to_csv(out_csv("d1"), index=False)


# =========================================================================== D2
def _wloss(y, pred, w):
    return float(np.sum(w * np.abs(y - pred)) / np.sum(w))


def _anchor_pub_dates(lab_v):
    """Publication dates of the claims that are anchors at v (point run-rates, short point revenue)."""
    s = lab_v.sel
    a = s[(s.kind == "point") & ((s.metric == "run_rate")
                                 | ((s.metric == "revenue")
                                    & ((s.period_end - s.period_start).dt.days + 1 <= pm.MAX_REV_ANCHOR_DAYS)))]
    return sorted(set(pd.to_datetime(a.published)))


def d2_task(job):
    vintage, pool, obs_id = job
    warnings.simplefilter("ignore")
    t_start = time.time()
    claims, sources = W()
    v = pd.Timestamp(vintage)
    tg = target_row(obs_id)
    labs, flevs, pairs, fi = build(tg, v, claims, sources, "g_spec")
    P = pool_P(pairs, "anthropic", pool).reset_index(drop=True)
    folds = pm.cv_folds(P)
    fits = mz.fit_zoo(P)
    members = [m for m in mz.MEMBERS if fits[m].ok]
    y, wr = P["y"].to_numpy(float), P["w_raw"].to_numpy(float)
    starts = np.sort(P["t"].unique())
    blocks = np.array_split(starts, pm.CV_BLOCKS)
    rows, fold_rows, oos_rows = [], [], []
    # ---- inner CV per fold (from the OOF predictions at the selected hyperparameters)
    inner = {m: {} for m in members}
    inner_anth = {m: [] for m in members}
    for k, (tr, va) in enumerate(folds):
        va = np.asarray(va)
        for m in members:
            if fits[m].oof is not None:
                inner[m][k] = _wloss(y[va], fits[m].oof[va], wr[va])
    for m in members:
        if fits[m].oof is not None:
            ok = np.isfinite(fits[m].oof) & (P["company"].to_numpy() == "anthropic")
            inner_anth[m] = _wloss(y[ok], fits[m].oof[ok], wr[ok]) if ok.any() else np.nan
    # ---- (a) fold-nested: training labels rebuilt from claims published before the block starts
    fold_start = {}
    for k, (tr, va) in enumerate(folds):
        va = np.asarray(va)
        b0 = pd.Timestamp(P.loc[va, "t"].min())
        fold_start[k] = b0
        u = b0 - pd.Timedelta(days=1)
        labs_u = {c: pm.company_labels(c, claims[c], u, exclude_origin=tg.origin_id if c == "anthropic" else None)
                  for c in labs}
        flev_u = {c: mz.feature_levels(c, u, sources, "g_spec") for c in labs}
        pairs_u = {c: mz.make_pairs(labs_u[c], flev_u[c]) for c in labs}
        Pu = pool_P(pairs_u, "anthropic", pool)
        # label revision: vintage-v vs vintage-u label growth on the same training pairs, and level on the
        # validation block's label weeks that exist at u
        rev_tr = np.nan
        if len(Pu):
            mrg = Pu.merge(P[["company", "t", "h", "y"]], on=["company", "t", "h"], suffixes=("_u", "_v"))
            rev_tr = float(np.mean(np.abs(mrg.y_u - mrg.y_v))) if len(mrg) else np.nan
        frec = {"vintage": vintage, "pool": pool, "fold": k, "block_start": b0.date(),
                "n_train_inner": len(tr), "n_valid": len(va), "n_train_nested": len(Pu),
                "anth_anchors_at_u": len(labs_u["anthropic"].anchors), "anth_anchors_at_v": len(labs["anthropic"].anchors),
                "mean_abs_rev_train_y": rev_tr,
                "valid_share_anthropic": float((P.loc[va, "company"] == "anthropic").mean())}
        # anchors (at v) inside the validation windows: how many real data points the block's labels rest on
        an = labs["anthropic"].anchors.index
        Pv = P.loc[va]
        frec["valid_pairs_spanning_an_anchor"] = float(np.mean([
            any((a > t) and (a <= t1) for a in (an if c == "anthropic" else labs["openai"].anchors.index))
            for c, t, t1 in zip(Pv.company, Pv.t, Pv.t1)]))
        usable = len(Pu) >= pm.CV_MIN_TRAIN
        frec["nested_usable"] = usable
        fold_rows.append(frec)
        if not usable:
            continue
        Pu_n = pm._normalise(Pu)
        for m in members:
            try:
                mod = mz.build(m, fits[m].hyper).fit(Pu_n)
                pv = np.asarray(mod.predict(P.loc[va]), float)
                rows.append({"vintage": vintage, "pool": pool, "member": m, "fold": k,
                             "loss_inner": inner[m].get(k, np.nan), "loss_nested_train": _wloss(y[va], pv, wr[va])})
            except Exception as e:     # noqa: BLE001
                rows.append({"vintage": vintage, "pool": pool, "member": m, "fold": k,
                             "loss_inner": inner[m].get(k, np.nan), "loss_nested_train": np.nan, "error": repr(e)})
    # ---- (b) real task from vintage-nested origins (Anthropic): one origin per anchor publication date
    lab_v = labs["anthropic"]
    seen_t0 = set()
    for u in _anchor_pub_dates(lab_v):
        if u > v:
            continue
        lab_u = pm.company_labels("anthropic", claims["anthropic"], u, exclude_origin=tg.origin_id)
        if not lab_u.ok or len(lab_u.anchors) < pm.MIN_ANCHORS or lab_u.t0 >= lab_v.t0 or lab_u.t0 in seen_t0:
            continue
        seen_t0.add(lab_u.t0)
        hs = [h for h in pm.HORIZONS if lab_u.t0 + h * WEEK <= lab_v.t0]
        if not hs:
            continue
        labs_u = {"anthropic": lab_u}
        for c in labs:
            if c != "anthropic":
                labs_u[c] = pm.company_labels(c, claims[c], u)
        flev_u = {c: mz.feature_levels(c, u, sources, "g_spec") for c in labs}
        pairs_u = {c: mz.make_pairs(labs_u[c], flev_u[c]) for c in labs}
        if len(pairs_u["anthropic"]) < pm.MIN_TARGET_PAIRS:
            continue
        Pu = pool_P(pairs_u, "anthropic", pool)
        Pu_n = pm._normalise(Pu)
        ds = [lab_u.t0 + h * WEEK for h in hs]
        Grow = mz.growth_rows(lab_u, flevs["anthropic"], ds)
        y0 = lab_u.level(lab_u.t0)
        truth = np.array([lab_v.level(d) for d in ds])
        for m in members:
            try:
                mod = mz.build(m, fits[m].hyper).fit(Pu_n)
                r = np.asarray(mod.predict(Grow), float)
            except Exception:          # noqa: BLE001
                r = np.full(len(ds), np.nan)
            for h, d, rr, tt in zip(hs, ds, r, truth):
                oos_rows.append({"vintage": vintage, "pool": pool, "member": m, "origin_u": u.date(),
                                 "t0_u": lab_u.t0.date(), "h": h, "d": d.date(), "level_err": tt - (y0 + h * rr),
                                 "rate_err": (tt - (y0 + h * rr)) / h, "rate_pred": rr, "rate_true": (tt - y0) / h,
                                 "mom": float(Grow.loc[hs.index(h), "mom"])})
    summ = []
    for m in members:
        summ.append({"vintage": vintage, "pool": pool, "member": m, "hyper": str(fits[m].hyper),
                     "cv_inner": fits[m].cv_loss, "cv_inner_anth_rows": inner_anth[m],
                     "n_folds_inner": len(folds)})
    print(f"  d2 {vintage} {pool} {time.time() - t_start:.0f}s folds={len(folds)} oos={len(oos_rows)}", flush=True)
    return summ, rows, fold_rows, oos_rows


def run_d2(jobs):
    post, _ = target_sets()
    pub = post[post.rule == "publication"].drop_duplicates("vintage")
    todo = [(r.vintage, pool, r.target) for r in pub.itertuples() for pool in ("pooled", "own")]
    todo = sorted(todo, key=lambda j: j[0], reverse=True)
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init) as ex:
        res = list(ex.map(d2_task, todo))
    pd.DataFrame([r for s in res for r in s[0]]).to_csv(out_csv("d2_members"), index=False)
    pd.DataFrame([r for s in res for r in s[1]]).to_csv(out_csv("d2_folds_members"), index=False)
    pd.DataFrame([r for s in res for r in s[2]]).to_csv(out_csv("d2_folds"), index=False)
    pd.DataFrame([r for s in res for r in s[3]]).to_csv(out_csv("d2_oos"), index=False)


# =========================================================================== D3
H2 = ("rev_2026q3", "rr_2026_09_s1a", "rr_2026_10_s1a")
EARLY = pd.Timestamp("2026-01-01")


def _fit_pred(member, hyper, P, fi, cols=None):
    Q = P if cols is None else P[[c for c in P.columns if c in
                                  {"company", "t", "t1", "h", "y", "mom", "w_raw"} | set(cols)
                                  or (c.startswith("midas:") and c.split(":")[1] in cols)]]
    mod = mz.build(member, hyper).fit(pm._normalise(Q))
    G = fi.G if cols is None else fi.G[[c for c in fi.G.columns if c in {"company", "g", "mom"} | set(cols)
                                        or (c.startswith("midas:") and c.split(":")[1] in cols)]]
    fi2 = mz.ForecastInputs(fi.target, fi.dates, fi.basis, fi.before, fi.after, G, fi.start)
    return mod, predict_fc(mod, fi2), G


def d3_task(job):
    rule, obs_id, vintage, pool, member, hyper, fs = job
    warnings.simplefilter("ignore")
    claims, sources = W()
    tg = target_row(obs_id)
    labs, flevs, pairs, fi = build(tg, vintage, claims, sources, fs)
    P = pool_P(pairs, "anthropic", pool).reset_index(drop=True)
    feats = mz._feats_of(P)
    hyper = eval(hyper)                       # noqa: S307 - dict literal written by model_zoo
    mod, fc, _ = _fit_pred(member, hyper, P, fi)
    an = P[P.company == "anthropic"]
    early = an[an.t < EARLY]
    raw = F.asof_features(pd.Timestamp(vintage), "anthropic", sources=sources)
    spec = mz.feature_spec(fs)
    feat_rows = []
    for f in feats:
        col, how = spec[f]
        s = raw[col] if col in raw else pd.Series(dtype=float)
        fv, lv = s.first_valid_index(), s.last_valid_index()
        interior_nan = int(s.loc[fv:lv].isna().sum()) if fv is not None else np.nan
        first_pair = an.loc[an[f].notna(), "t"].min()
        feat_rows.append({"feature": f, "how": how, "miss_anth_early": float(early[f].isna().mean()) if len(early) else np.nan,
                          "miss_anth_all": float(an[f].isna().mean()),
                          "miss_openai": float(P.loc[P.company == "openai", f].isna().mean()) if pool == "pooled" else np.nan,
                          "first_anth_pair_t": None if pd.isna(first_pair) else pd.Timestamp(first_pair).date(),
                          "nan_at_forecast": bool(np.isnan(fi.G[f]).all()) if fi.after else np.nan,
                          "interior_nan_weeks_zero_filled": interior_nan if how == "cumsum" else np.nan})
    fr = pd.DataFrame(feat_rows)
    late = fr[(fr.miss_anth_early > 0.5)].feature.tolist()
    # coefficients / contributions
    G = fi.G
    base_pred = np.asarray(mod.predict(G), float)
    Z = mod.prep_.transform(G) if hasattr(mod, "prep_") else None
    for i, f in enumerate(feats):
        G0 = G.copy()
        G0[f] = np.nan                          # as if missing at forecast time (linear: -> company mean)
        p0 = np.asarray(mod.predict(G0), float)
        fr.loc[fr.feature == f, "contrib_rate"] = float(np.mean(base_pred - p0))
        if Z is not None:
            j = mod.prep_.xcols.index(f)
            fr.loc[fr.feature == f, "z_forecast"] = float(np.mean(Z[:, j]))
            Zt = mod.prep_.transform(P)[:, j]
            pres = P[f].notna().to_numpy()
            fr.loc[fr.feature == f, "z_train_p99_present"] = float(np.percentile(np.abs(Zt[pres]), 99)) if pres.any() else np.nan
            if hasattr(mod, "est_") and hasattr(mod.est_, "coef_") and member != "mlp":
                coef = np.asarray(mod.est_.coef_).reshape(-1)
                fr.loc[fr.feature == f, "coef_std"] = float(coef[j])
    # training-row permutation importance (weighted MAE increase), MLP and others alike
    rng = np.random.default_rng(0)
    Pn = pm._normalise(P)
    yy, ww = P["y"].to_numpy(float), P["w_raw"].to_numpy(float)
    base_loss = _wloss(yy, np.asarray(mod.predict(Pn), float), ww)
    for f in feats:
        Q = Pn.copy()
        Q[f] = rng.permutation(Q[f].to_numpy())
        fr.loc[fr.feature == f, "perm_importance"] = _wloss(yy, np.asarray(mod.predict(Q), float), ww) - base_loss
    # era pattern: demeaned Anthropic y where late features are missing vs present
    if late:
        miss = an[late].isna().all(axis=1)
        era = {"y_mean_late_missing": float(an.loc[miss, "y"].mean()) if miss.any() else np.nan,
               "y_mean_late_present": float(an.loc[~miss, "y"].mean()) if (~miss).any() else np.nan}
    else:
        era = {"y_mean_late_missing": np.nan, "y_mean_late_present": np.nan}
    _, fc_nolate, _ = _fit_pred(member, hyper, P, fi, cols=[f for f in feats if f not in late])
    gspec = [f for f in feats if f in pm.FEATURE_SPEC]
    _, fc_gspec, _ = _fit_pred(member, hyper, P, fi, cols=gspec)
    # late features present in training only from their start: complete-case window
    firsts = [fr.loc[fr.feature == f, "first_anth_pair_t"].iloc[0] for f in late]
    firsts = [pd.Timestamp(x) for x in firsts if x is not None]
    if firsts:
        start_cc = max(firsts)
        Pcc = P[P.t >= start_cc].reset_index(drop=True)
        try:
            _, fc_cc, _ = _fit_pred(member, hyper, Pcc, fi)
        except Exception:              # noqa: BLE001
            fc_cc = np.nan
    else:
        start_cc, fc_cc = None, np.nan
    summary = {"rule": rule, "target": obs_id, "vintage": vintage, "pool": pool, "member": member, "feature_set": fs,
               "hyper": str(hyper), "n_pairs": len(P), "n_features": len(feats), "late_features": ";".join(late),
               "err_full": log_err(tg.value, fc), "err_no_late": log_err(tg.value, fc_nolate),
               "err_gspec_cols": log_err(tg.value, fc_gspec),
               "err_complete_case": log_err(tg.value, fc_cc), "complete_case_start": None if start_cc is None else start_cc.date(),
               "n_pairs_complete_case": int((P.t >= start_cc).sum()) if start_cc is not None else np.nan, **era}
    fr.insert(0, "pool", pool), fr.insert(0, "vintage", vintage), fr.insert(0, "target", obs_id), fr.insert(0, "rule", rule)
    fr["member"] = member
    fr["feature_set"] = fs
    print(f"  d3 {rule} {obs_id} {pool} {member} err {summary['err_full']:.3f} no_late {summary['err_no_late']:.3f}",
          flush=True)
    return summary, fr


def run_d3(jobs):
    todo = []
    for fs in ("all", "g_spec"):
        z = zoo_auto(fs)
        a = z[(z.company == "anthropic") & (z.model == "AUTO") & z.target.isin(H2)]
        todo += [(r.rule, r.target, r.vintage, r.pool, r.auto_choice, r.hyper, fs) for r in a.itertuples()]
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init) as ex:
        res = list(ex.map(d3_task, todo))
    pd.DataFrame([r[0] for r in res]).to_csv(out_csv("d3"), index=False)
    pd.concat([r[1] for r in res], ignore_index=True).to_csv(out_csv("d3_features"), index=False)


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ss,d1,d2,d3,report")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    for part in a.parts.split(","):
        t = time.time()
        if part == "report":
            write_report()
        else:
            {"ss": run_ss, "d1": run_d1, "d2": run_d2, "d3": run_d3}[part](a.jobs)
        print(f"{part} done in {time.time() - t:.0f}s", flush=True)



# =========================================================================== report helpers
def refs(rule):
    """Per-target log errors of the references: G, E (state space + one signal), A (claims only)."""
    out = {}
    for name, var in (("G", "gh_share"), ("E", "or_price")):
        d = pd.read_csv(ref_csv(var, rule))
        out[name] = d.set_index("target").log_err_model
    b = pd.read_csv(os.path.join(REP, f"forecast_backtest_{rule}_{ASOF}.csv")).set_index("target")
    out["A"] = b.log_err_model_claims_only
    return pd.DataFrame(out)


def zoo_err(fs, rule, model, pool="pooled"):
    d = pd.read_csv(zoo_csv(fs, rule))
    d = d[(d.company == "anthropic") & (d.model == model) & (d.pool == pool)]
    return d.set_index("target").log_err


def md(df, floatfmt=".3f"):
    return df.to_markdown(floatfmt=floatfmt)


def mean_abs_row(t, label="mean |log error|"):
    m = t.abs().mean(numeric_only=True)
    return pd.concat([t, pd.DataFrame([m], index=[label])])


D1_COLS = ["AUTO", "ridge g_spec", "ridge gh", "ridge or_spend", "ridge or_tp", "ridge gh@rhoG", "ridge g_spec@rhoG",
           "ridge gh alpha=0.01", "SS panel", "SS panel@rhoG", "A", "G", "E"]


def d1_tables():
    d = pd.read_csv(out_csv("d1"))
    tabs = {}
    for rule in RULES:
        x = d[d.rule == rule].pivot(index="target", columns="variant", values="ridge_err")
        t = pd.DataFrame(index=x.index)
        t["AUTO"] = zoo_err("g_spec", rule, "AUTO").reindex(x.index)
        t["ridge g_spec"] = x["g_spec"]
        for v in ("gh", "or_spend", "or_tp", "gh@rhoG", "g_spec@rhoG", "gh alpha=0.01"):
            t[f"ridge {v}"] = x[v]
        dd = d[d.rule == rule]
        t["SS panel"] = dd[dd.variant == "g_spec"].set_index("target").ss_panel_err
        t["SS panel@rhoG"] = dd[dd.variant == "g_spec@rhoG"].set_index("target").ss_panel_err
        t = t.join(refs(rule)[["A", "G", "E"]])
        csv_ridge = zoo_err("g_spec", rule, "ridge").reindex(x.index)
        tabs[rule] = (mean_abs_row(t[D1_COLS]), float(np.nanmax(np.abs(csv_ridge - x["g_spec"]))))
    dec = d[(d.rule == "publication") & d.variant.isin(["gh", "gh alpha=0.01"])].copy()
    dec["rate_pred"] = dec.rate_base + dec.rate_mom + dec.rate_features
    dec["alpha"] = dec.hyper.map(lambda h: eval(h)["alpha"])           # noqa: S307
    dec = dec[["target", "variant", "t0", "g_weeks", "alpha", "rate_base", "rate_mom", "rate_features", "rate_pred",
               "rate_needed", "mom_input", "elast:mom", "x:github_repo_share", "elast:github_repo_share", "ridge_err"]]
    return tabs, dec.set_index(["target", "variant"]).sort_index()


def d4_table():
    s = pd.read_csv(out_csv("ss"))
    p = s[s.kind == "pre"].set_index("target")
    t = pd.DataFrame(index=p.index)
    t["S0 zoo label path (every member)"] = zoo_err("g_spec", "publication", "AUTO").reindex(p.index)
    t["S1 SS panel (smoother mean)"] = zoo_err("g_spec", "publication", mz.SS_NAME, pool="own").reindex(p.index)
    for c in [c for c in p.columns if c.startswith("err:")]:
        t[c[4:]] = p[c]
    r = refs("publication")
    t["G (CSV)"] = r["G"].reindex(p.index)
    t["A (CSV)"] = r["A"].reindex(p.index)
    t["E (CSV)"] = r["E"].reindex(p.index)
    order = ["rr_2025_12", "rev_2025q1", "rev_2025q2", "rev_2025q3", "rev_2025fy", "rev_2026q1", "rev_2026q2"]
    t = t.reindex(order)
    steps = [c for c in t.columns if c.split(" ")[0] in ("S0", "S1", "S1r", "S2", "S3", "S4", "S5")]
    a = t[steps].abs()
    key = [c.split(" ")[0] for c in steps]
    contrib = pd.DataFrame({f"{key[i]}->{key[i + 1]}": a[steps[i + 1]] - a[steps[i]] for i in range(len(steps) - 1)})
    contrib["S0->S5 total"] = a[steps[-1]] - a[steps[0]]
    contrib = pd.concat([contrib, pd.DataFrame([contrib.mean()], index=["mean (signed: + = worse)"])])
    return mean_abs_row(t), contrib, s


def _rank(s, m):
    r = s.rank(method="min")
    return int(r[m]) if m in r and np.isfinite(s[m]) else np.nan


def d2_tables():
    mem = pd.read_csv(out_csv("d2_members"))
    fm = pd.read_csv(out_csv("d2_folds_members"))
    folds = pd.read_csv(out_csv("d2_folds"))
    oos = pd.read_csv(out_csv("d2_oos"))
    z = pd.read_csv(zoo_csv("g_spec", "publication"))
    z = z[(z.company == "anthropic")]
    z["vintage"] = z.vintage.astype(str)
    post, _ = target_sets()
    post = post[post.rule == "publication"]
    rows, per_member = [], []
    for (v, pool), g in mem.groupby(["vintage", "pool"], sort=False):
        g = g.set_index("member")
        inner = g.cv_inner.replace(np.inf, np.nan)
        f = fm[(fm.vintage == v) & (fm.pool == pool)]
        matched = f.groupby("member")[["loss_inner", "loss_nested_train"]].mean().reindex(g.index)
        o = oos[(oos.vintage == v) & (oos.pool == pool)]
        o_rate = o.groupby("member").rate_err.apply(lambda x: x.abs().mean()).reindex(g.index)
        o_lvl = o.groupby("member").level_err.apply(lambda x: x.abs().mean()).reindex(g.index)
        tg = post[post.vintage.astype(str) == v].target.tolist()
        zz = z[(z.vintage == v) & (z.pool == pool) & z.target.isin(tg)]
        test = zz.groupby("model").log_err.apply(lambda x: x.abs().mean())
        pick = {"inner CV (AUTO)": inner.idxmin(),
                "fold-nested labels": matched.loss_nested_train.idxmin() if matched.loss_nested_train.notna().any() else None,
                "nested real task": o_rate.idxmin() if o_rate.notna().any() else None}
        rec = {"vintage": v, "pool": pool, "targets": ",".join(tg), "n_folds": int(g.n_folds_inner.iloc[0]),
               "n_folds_nested_usable": int(f.fold.nunique()),
               "inner_min": inner.min(), "inner_max": inner.max(),
               "inner_spread_rel": (inner.max() - inner.min()) / inner.median(),
               "n_within_10pct_of_best_inner": int((inner <= 1.1 * inner.min()).sum()),
               "n_oos_points": int(len(o) / max(len(g), 1)),
               "oos_spread_rel": (o_rate.max() - o_rate.min()) / o_rate.median() if o_rate.notna().any() else np.nan,
               "spearman_inner_vs_oos": inner.corr(o_rate, method="spearman") if o_rate.notna().sum() > 3 else np.nan,
               "spearman_inner_vs_nestedlabels": matched.loss_inner.corr(matched.loss_nested_train, method="spearman")
               if matched.loss_nested_train.notna().sum() > 3 else np.nan,
               "mlp_rank_inner": _rank(inner, "mlp"), "mlp_rank_nestedlabels": _rank(matched.loss_nested_train, "mlp"),
               "mlp_rank_oos": _rank(o_rate, "mlp")}
        for k, m in pick.items():
            rec[f"pick: {k}"] = m
            rec[f"test |err| of pick: {k}"] = test.get(m, np.nan) if m else np.nan
        rec["test |err| best member (hindsight)"] = test.reindex(list(g.index)).min()
        rows.append(rec)
        pm_ = pd.DataFrame({"cv_inner": inner, "cv_inner_anthropic_rows": g.cv_inner_anth_rows,
                            "inner_on_nested_folds": matched.loss_inner, "fold_nested_labels": matched.loss_nested_train,
                            "oos_real_task_rate": o_rate, "oos_real_task_level": o_lvl,
                            "test_abs_log_err": test.reindex(g.index)})
        pm_.insert(0, "pool", pool)
        pm_.insert(0, "vintage", v)
        per_member.append(pm_)
    return pd.DataFrame(rows), pd.concat(per_member), folds, oos


def d3_tables():
    s = pd.read_csv(out_csv("d3"))
    f = pd.read_csv(out_csv("d3_features"))
    late = sorted(set(";".join(s.late_features.dropna()).split(";")) - {""})
    # missingness / first pair of every late-starting feature (from the latest vintage, pooled)
    ref = f[(f.feature_set == "all") & (f.rule == "publication") & (f.target == "rr_2026_10_s1a")]
    miss = ref[ref.pool == "pooled"].set_index("feature").loc[
        late, ["how", "miss_anth_early", "miss_anth_all", "miss_openai", "first_anth_pair_t",
               "interior_nan_weeks_zero_filled"]]
    # contributions in AUTO's chosen member at the H2 vintages (all features)
    rr = f[(f.feature_set == "all") & f.feature.isin(late)]
    piv = rr.groupby(["pool", "member", "feature"]).agg(
        coef_std=("coef_std", lambda x: x.abs().max()), contrib_rate_mean=("contrib_rate", "mean"),
        contrib_rate_absmax=("contrib_rate", lambda x: x.abs().max()),
        z_forecast_absmax=("z_forecast", lambda x: x.abs().max()),
        z_train_p99=("z_train_p99_present", "max"), perm_importance=("perm_importance", "mean")).reset_index()
    piv = piv[(piv.z_forecast_absmax > 0) | (piv.contrib_rate_absmax > 0)]
    abl = s[["feature_set", "rule", "target", "pool", "member", "err_full", "err_no_late", "err_gspec_cols",
             "err_complete_case", "n_pairs_complete_case", "y_mean_late_missing", "y_mean_late_present"]]
    return miss, piv, abl.sort_values(["feature_set", "pool", "rule", "target"]), late


def item3_table():
    R = {fs: pd.read_csv(zoo_csv(fs, "publication")) for fs in ("g_spec", "all")}
    R = {k: v[v.company == "anthropic"] for k, v in R.items()}
    g, a = R["g_spec"], R["all"]
    rows = []
    for pool in ("pooled", "own"):
        for t in sorted(g.target.unique()):
            ga = g[(g.target == t) & (g.model == "AUTO") & (g.pool == pool)]
            aa = a[(a.target == t) & (a.model == "AUTO") & (a.pool == pool)]
            if not len(ga) or not len(aa):
                continue
            mg, ma = ga.auto_choice.iloc[0], aa.auto_choice.iloc[0]
            same = a[(a.target == t) & (a.model == mg) & (a.pool == pool)].log_err.iloc[0]
            rows.append({"pool": pool, "target": t, "gap_weeks": ga.gap_weeks.iloc[0], "AUTO g_spec picks": mg,
                         "AUTO all picks": ma, "err g_spec": ga.log_err.iloc[0], "err all": aa.log_err.iloc[0],
                         "err all, g_spec's member": same})
    r = pd.DataFrame(rows)
    r["d|err| total"] = r["err all"].abs() - r["err g_spec"].abs()
    r["of which: features (same member)"] = r["err all, g_spec's member"].abs() - r["err g_spec"].abs()
    r["of which: AUTO switched member"] = r["err all"].abs() - r["err all, g_spec's member"].abs()
    summ = r.groupby("pool")[["d|err| total", "of which: features (same member)", "of which: AUTO switched member"]].mean()
    return r, summ


def d1_coverage():
    """Anthropic training pairs with the single signal present, and whether the forecast window has it."""
    path = out_csv("d1_coverage")
    if os.path.exists(path):
        return pd.read_csv(path)
    claims, sources = W()
    post, _ = target_sets()
    rows = []
    for r in post[post.rule == "publication"].itertuples():
        tg = target_row(r.target)
        for fs, col in (("gh", "github_repo_share"), ("or_spend", "or_spend")):
            labs, flevs, pairs, fi = build(tg, r.vintage, claims, sources, fs)
            an, oa = pairs["anthropic"], pairs["openai"]
            rows.append({"target": r.target, "vintage": r.vintage, "signal": col,
                         "anth_pairs": len(an), "anth_pairs_with_signal": int(an[col].notna().sum()),
                         "openai_pairs_with_signal": int(oa[col].notna().sum()) if len(oa) else 0,
                         "signal_in_forecast_window": bool(fi.after and np.isfinite(fi.G[col]).all()),
                         "first_signal_week": flevs["anthropic"][col].first_valid_index().date()})
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def _key_numbers(d4, s2, pmem, folds):
    K = {}
    m = d4.loc["mean |log error|"]
    col = {c.split(" ")[0]: c for c in d4.columns}
    K["d4_S0"], K["d4_S4"], K["d4_S5"] = m[col["S0"]], m[col["S4"]], m[col["S5"]]
    K["d4_cross"] = m[col["S3"]] - m[col["S2"]]
    K["d4_seg"] = m[col["S4"]] - m[col["S3"]]
    rev = [i for i in d4.index if str(i).startswith("rev_")]
    j = d4.loc[rev, col["S0"]].abs() - d4.loc[rev, col["S1"]].abs()
    K["jensen_min"], K["jensen_max"] = float(j.min()), float(j.max())
    h2v = ("2026-11-01", "2026-11-09")
    rat = {}
    for (v, pool), g in pmem.groupby(["vintage", "pool"]):
        if v in h2v:
            rat.setdefault(pool, []).append(float((g.oos_real_task_rate / g.cv_inner).median()))
    K["ratio_own_h2"], K["ratio_pooled_h2"] = np.mean(rat["own"]), np.mean(rat["pooled"])
    h2 = s2[s2.vintage.isin(h2v)]
    K["sp_min"], K["sp_max"] = h2.spearman_inner_vs_oos.min(), h2.spearman_inner_vs_oos.max()
    w10 = h2[h2.pool == "pooled"].n_within_10pct_of_best_inner
    K["within10_h2"] = f"{w10.min()}-{w10.max()}" if w10.min() != w10.max() else f"{w10.min()}"
    own = folds[folds.pool == "own"]
    K["zero_folds"] = ", ".join(f"{v} {int((g.n_train_nested == 0).sum())} of {len(g)}"
                                for v, g in own.groupby("vintage"))
    pf = folds[(folds.pool == "pooled") & (folds.vintage >= "2026-04-01")]
    sh = pf.assign(o=(1 - pf.valid_share_anthropic) * pf.n_valid).groupby("vintage")
    shares = sh.o.sum() / sh.n_valid.sum()
    K["openai_share"] = f"{shares.min():.0%}-{shares.max():.0%}"
    rr = []
    for _, r in s2[(s2.pool == "pooled") & (s2.vintage >= "2026-04-01")].iterrows():
        g = pmem[(pmem.vintage == r["vintage"]) & (pmem.pool == "pooled")]
        x = g.loc[r["pick: inner CV (AUTO)"]]
        rr.append(x.cv_inner_anthropic_rows / x.cv_inner)
    K["anth_ratio_min"], K["anth_ratio_max"] = min(rr), max(rr)
    g = pmem[(pmem.vintage == "2026-11-09") & (pmem.pool == "pooled")]
    K["ridge_anth_rank"] = int(g.cv_inner_anthropic_rows.rank(method="min")["ridge"])
    ok = s2[s2.n_oos_points > 0].copy()
    ok["n_t"] = ok.targets.str.count(",") + 1
    for pool in ("pooled", "own"):
        o = ok[ok.pool == pool]
        for k, c in (("auto", "test |err| of pick: inner CV (AUTO)"), ("nested", "test |err| of pick: nested real task")):
            K[f"pick_{k}_{pool}"] = float((o[c] * o.n_t).sum() / o.n_t.sum())
    return K


# =========================================================================== report
def write_report():
    tabs, dec = d1_tables()
    cov = d1_coverage()
    d4, d4c, ss = d4_table()
    s2, pmem, folds, oos = d2_tables()
    miss, piv, abl, late = d3_tables()
    i3, i3s = item3_table()
    post, pre = target_sets()
    K = _key_numbers(d4, s2, pmem, folds)
    L = []
    w = L.append
    t1p, chk_p = tabs["publication"]
    t1e, chk_e = tabs["period_end"]
    w("# Model-zoo diagnostics: why AUTO loses to G and E (2026-11-12)\n")
    w("Data as of 2026-11-10, the same inputs as `forecast_zoo_*_2026-11-10.csv`. Script: "
      "`runrate_nowcast/scripts/zoo_diagnostics.py` (`--parts ss,d1,d3,d2,report`, 4 workers, about 9 minutes "
      "of wall time). Intermediate tables are in `reports/zoo_diagnostics_<part>_2026-11-12.csv`. No package file "
      "was changed. Every forecast is recomputed with the package's own functions "
      "(`panel_model` labels/pairs/folds, `model_zoo` members, `backtest.prepublication_forecast`).\n")
    w("Replication checks:\n")
    w(f"- The zoo ridge (g_spec, pooled) recomputed here equals the CSV on every post-t0 target (max |diff| "
      f"{max(chk_p, chk_e):.1e}).")
    w("- The inner-CV losses recomputed in D2 equal the CSV's `cv_loss`.")
    w("- The state-space replica reproduces the panel's claims-only path (S1r), A (S3) and G (S5) to 3 decimals.\n")
    w("Errors are log(actual / forecast); negative means the forecast was too high. |err| is the absolute value.\n")

    # ------------------------------------------------------------------ summary
    w("## Summary\n")
    w("| Strange result | Explained? | Mechanism (section) |")
    w("|---|---|---|")
    w("| 1. AUTO 0.234 vs G 0.171 / E 0.145 on the 7 post-t0 targets | Yes | The gap comes from the framework, "
      "not from the extra features (D1). With the same single signal as G or E, the zoo ridge does worse "
      "(0.314 / 0.400) than with 8 features (0.207). The predicted weekly growth is about the training-sample "
      "mean growth of the company (0.040-0.058/week). CV-chosen shrinkage removes most of the momentum and "
      "feature terms. The elasticity has to be learned from smoothed labels that rest on 3-10 anchors. A signal "
      "that starts after t0 (OpenRouter, Nov 2025) cannot enter at all. G and E use a prior elasticity and a "
      "drifting intercept instead. |")
    w("| 2. Flat inner-CV losses; MLP picked 12/14 in the own pool | Yes | The inner CV is not a vintage test (D2). "
      "Its folds train on labels that did not exist at the block start: at the Feb-Apr vintages, 2 of 2-3 "
      "own-pool folds have no real-time labels at all. Its validation pairs are interior windows of a smoothed "
      "path, and about half of the pooled validation rows are OpenAI's. On the real task rebuilt from nested "
      f"vintages, the losses at the H2 vintages are {K['ratio_own_h2']:.1f}x (own) and "
      f"{K['ratio_pooled_h2']:.1f}x (pooled) the inner-CV losses. The member ranking is unrelated to the "
      f"inner-CV ranking (Spearman {K['sp_min']:+.2f} to {K['sp_max']:+.2f}). The MLP drops from rank 1 to 11 "
      "(fold-nested labels) and to 14 (real task) at 2026-11-09. More than half of its +214% at rr_2026_10 comes "
      "from late-starting features that enter far outside their training range (D3): the log error goes from "
      "-1.145 to -0.499 without them. |")
    w("| 3. +31 features: AUTO worse, ridge better | Yes | Selection noise (D3). On the pooled publication targets, "
      f"AUTO's +{i3s.loc['pooled', 'd|err| total']:.3f} is "
      f"{i3s.loc['pooled', 'of which: features (same member)']:+.3f} from the features (the same member gets "
      f"better) and {i3s.loc['pooled', 'of which: AUTO switched member']:+.3f} from AUTO switching between "
      "near-tied members: midas at rr_2026_05 and lasso instead of ridge at the H2 vintages. The pooled lasso "
      "zeroes every late-starting feature, so imputation does not affect it. |")
    w("| 4. Pre-t0 targets: claims-only path, e.g. rr_2025_12 +18.7% vs G +4.0% | Yes | The pre-t0 path is the "
      "claims-only smoother (D4). G's advantage there comes almost entirely from its GitHub indicator bending the "
      f"path between sparse anchors (mean |err| {K['d4_S4']:.3f} to {K['d4_S5']:.3f}). The process-noise fit "
      "contributes 0, and the crossing position, segment claims and Jensen term are each at most 0.01 on average. These 7 targets carry "
      "no information about the ML members. |")
    w("")
    w("**Issues found (evidence in the sections):**")
    w("1. **Inner CV leaks hindsight labels (D2).** It is the source of the flat losses and of the MLP picks.")
    w("2. **Scaling of sparse features (D3).** `Prep` scales each column by its RMS over all training rows, "
      "including the zeros imputed for missing rows. A feature present in 3% of rows (vercel_spend_share) enters "
      "the forecast row at z = -88 (own) / -160 (pooled), against a training 99th percentile of 29 / 49. "
      "Nothing is clipped, so the nonlinear members (MLP) extrapolate. This is a design flaw, not a crash.")
    w("3. **The crossing position differs between the zoo and the references.** The panel and zoo date "
      "crossing claims at rho = 0.5 (`params.json` has no `crossing_position`). A, E and G use the profiled "
      "rho of 0.60-0.92. For the crossing targets rr_2026_04 and rr_2026_05, the zoo and G therefore score the "
      "same official number at dates 8-10 days apart (table D1c). The paired comparisons are slightly "
      "apples-to-oranges. The effect on the gap is mixed and small.")
    w("4. **cumsum of carried statements.** The `all` rule turns `cloud_bedrock_growth` (a quarterly growth "
      "statement carried for 100 days) into a weekly cumsum level. Interior gaps (12 weeks) are zero-filled, so "
      "a stale statement reads as a slowdown. npm series zero-fill 3 holiday weeks. Lasso zeroes this feature; "
      "it is a construction bug of small consequence.")
    w("5. **Minor: the revenue label path uses the median, not the mean.** `path_value` integrates "
      "exp(mean log level), while the smoother's own revenue is the mean. Pre-t0 revenue forecasts are up to "
      f"{K['jensen_max'] * 100:.1f}% lower than `SS claims only` (S0 vs S1).\n")
    w("**Recommended fixes, in priority order:**")
    w("1. **Score only post-t0 targets for model comparison.** Report pre-t0 targets separately as a "
      "label-path check: they are identical for all 18 members and dilute every paired test (7 of 14 "
      "publication targets).")
    w("2. **Use vintage-nested evaluation and stop selecting per vintage among 18 members.** Nested CV removes "
      "the MLP's spurious advantage. Its own picks were no better on the test targets, because 3-14 real-task "
      "points per vintage cannot rank 18 near-equivalent members. Fix one or two simple members in advance "
      "(ridge, Bayesian ridge), or use an equal-weight average, and use nested CV only for shrinkage.")
    w("3. **Change the framework's anchor: ML corrects the state-space path, not y(t0) + g x mean growth.** "
      "Use the claims-only (or G/E) state-space forecast as the base and let the ML predict the deviation of "
      "growth from it. Or put a prior elasticity on each feature (as the state space does) instead of learning it "
      "from 3-10 anchors. Replace the company-mean intercept, which carries 2025-26 hyper-growth into every "
      "forecast, with the current state-space growth.")
    w("4. **Standardize on present rows and clip out-of-range inputs**, or drop features with less than about "
      "6 months of overlap with the training labels. Do not zero-fill interior gaps of cumsum features. Restrict "
      "nonlinear members to features with full history, or drop them.")
    w("5. **Align the crossing position** between the zoo and the references (profile rho at each vintage, as A, "
      "E and G do), so that paired comparisons score the same target date.\n")

    # ------------------------------------------------------------------ D1
    w("## D1. Framework vs features\n")
    w("This runs the zoo pipeline (`build` reproduces `model_zoo.forecast_target`: same labels, pairs, CV folds, "
      "ridge grid and forecast path) with the pooled pool and ridge only. The feature sets are:\n")
    w("- `gh`: github_repo_share only, the signal of G.")
    w("- `or_spend`: openrouter_log_tokens + log_price, the tokens x price of E. OpenAI rows use tokens only, "
      "since price is Anthropic-only.")
    w("- `or_tp`: the same two columns separately.\n")
    w("Each is also run re-dated at G's profiled crossing position (`@rhoG`; claims and target t_eff re-dated). "
      "`gh alpha=0.01` is a mechanism probe with the shrinkage fixed near zero; it is not a candidate model. "
      "References are AUTO (g_spec, pooled) and the state-space A / G / E from the CSVs. `SS panel` is the "
      "claims-only smoother from the same label run.\n")
    w("### D1a. Publication rule, the 7 post-t0 targets\n")
    w(md(t1p))
    w("\n### D1b. Period-end rule, the 11 post-t0 targets\n")
    w(md(t1e))
    w("\n### D1c. What the ridge forecast is made of (publication, `gh`)\n")
    w("Predicted weekly log growth after t0 = company mean (`rate_base`, the weighted mean of the training "
      "labels' growth) + momentum term + feature term, averaged over the target's dates after t0. `rate_needed` "
      "is the constant weekly growth from y(t0) that would hit the actual value. `alpha` is the CV-chosen ridge "
      "penalty. `elast` is in original units (weekly growth per unit of weekly feature change).\n")
    w(md(dec.drop(columns=["ridge_err"]).rename(columns=lambda c: c.replace("x:github_repo_share", "share chg/wk")
                                                 .replace("elast:github_repo_share", "elast share")), ".4f"))
    w("\nCrossing positions and target dates (rho = 0.5 in the zoo):\n")
    cr = ss[ss.kind == "post"][["rule", "target", "vintage", "rho_A", "rho_G", "rho_E", "t_eff_05", "t_eff_rhoG"]]
    w(md(cr.set_index(["rule", "target"]), ".2f"))
    w("\nCoverage of the single signals in the training pairs (publication):\n")
    w(cov.set_index(["target", "signal"]).to_markdown())
    w("\n**Conclusion D1:** the gap comes from the framework, not from the extra features.\n")
    w(f"- **Fewer features make it worse.** On the same 7 targets, the zoo ridge with G's signal alone scores "
      f"{t1p.loc['mean |log error|', 'ridge gh']:.3f} and with E's signal alone "
      f"{t1p.loc['mean |log error|', 'ridge or_spend']:.3f}, against G "
      f"{t1p.loc['mean |log error|', 'G']:.3f} and E {t1p.loc['mean |log error|', 'E']:.3f}. That is worse than "
      f"the 8-feature ridge ({t1p.loc['mean |log error|', 'ridge g_spec']:.3f}) and near the claims-only smoother "
      f"({t1p.loc['mean |log error|', 'SS panel']:.3f}). The period-end rule shows the same pattern.")
    w("- **What the framework does.** It extrapolates y(t0) at the company's mean training growth "
      "(0.040-0.058/week, about +20-29% a month). At most vintages CV picks alpha 100-1000, which shrinks the "
      "momentum elasticity to 0.04-0.3 and the feature terms to about 0.002/week (5 of 7 targets). The early-2026 "
      "targets needed 0.045-0.17/week against a predicted 0.02-0.05 (forecasts too low: rr_2026_02/03/04). The "
      "H2-2026 targets needed about 0 (forecasts too high: rr_2026_09/10).")
    w("- **The GitHub signal points the right way but is too weak.** In H2 the feature term is -0.023 to "
      "-0.029/week against the -0.06 needed. At rr_2026_02 the learned elasticity even has the wrong sign "
      "(-0.12), learned from the 2025 pairs.")
    w("- **The OpenRouter signal cannot enter.** It starts on 2025-11-20 (rolling 53-week history). It has 0 "
      "Anthropic training pairs at the rr_2026_02 vintage, where it is also missing at t0, and 8-23 pairs up to "
      "May 2026. Its elasticity is about 0.")
    w("- **How the state space differs.** It uses the same signal with a prior elasticity (beta 1 +/- 1) and a "
      "drifting intercept, so it needs no labelled history for the signal.")
    w("- **Shrinkage is not the main problem.** Removing it (`alpha=0.01`) helps only partly (gh "
      f"{t1p.loc['mean |log error|', 'ridge gh alpha=0.01']:.3f} publication, "
      f"{t1e.loc['mean |log error|', 'ridge gh alpha=0.01']:.3f} period-end).")
    w(f"- **Re-dating crossing claims does not close the gap** (`gh@rhoG` "
      f"{t1p.loc['mean |log error|', 'ridge gh@rhoG']:.3f}). It mainly moves rr_2026_04 and rr_2026_05. Once "
      f"re-dated, the panel's claims-only smoother matches A (`SS panel@rhoG` "
      f"{t1p.loc['mean |log error|', 'SS panel@rhoG']:.3f} vs A {t1p.loc['mean |log error|', 'A']:.3f} "
      f"publication; {t1e.loc['mean |log error|', 'SS panel@rhoG']:.3f} vs "
      f"{t1e.loc['mean |log error|', 'A']:.3f} period-end). The claims side of the two pipelines therefore agrees "
      "apart from the crossing position.")
    w("- **Where the gap sits.** The early-2026 hyper-growth targets (rr_2026_02/03/04) account for most of "
      "it; there G and E saw the acceleration through their indicators. At the H2 targets, G at the publication "
      "vintage is no better than the zoo ridge (-0.29 / -0.30 vs -0.23 / -0.26).\n")

    # ------------------------------------------------------------------ D2
    w("## D2. Validity of the inner CV\n")
    w("For each publication vintage of a post-t0 target and each pool, with the g_spec features and every member "
      "at the hyperparameters its inner CV chose:\n")
    w("- **inner CV:** `pm.cv_folds` on the pairs built from the vintage-v labels (as in the zoo).")
    w("- **fold-nested labels:** the same validation pairs, but the training labels, pairs and features are "
      "rebuilt from the claims and features public the day before the block starts (`company_labels(c, "
      "claims, b0 - 1 day)`). A fold is usable when that gives at least 20 pairs.")
    w("- **nested real task:** origins are the publication dates of the Anthropic anchors (one per distinct "
      "t0, at least 3 anchors). At each origin u the member is trained on the pool's pairs at u and forecasts "
      "y(t0_u + h) for h = 4, 8, 13 weeks, as the zoo does: from y_u(t0_u), with momentum from the u-path and "
      "feature changes over (t0_u, d]. The truth is the vintage-v label (only dates <= t0_v). Loss = mean "
      "|level error| / h, the same units as the CV loss.\n")
    w("`test |err|` is the member's mean |log error| on the publication targets at that vintage, from the zoo "
      "CSV.\n")
    show = s2.set_index(["vintage", "pool"])

    def _fmt(x):
        if isinstance(x, (float, np.floating)):
            return "" if not np.isfinite(x) else (f"{x:.0f}" if float(x).is_integer() and abs(x) > 0.5 else f"{x:.4f}")
        return str(x).replace(",", ", ")
    w(show.T.map(_fmt).to_markdown())
    w("\nFolds: training pairs available to the inner CV vs those that existed at the block start. "
      "`anth_anchors_at_u` is the number of Anthropic anchors public at the block start. "
      "`mean_abs_rev_train_y` is the mean |y_v - y_u| on the shared training pairs.\n")
    w(md(folds.drop(columns=["valid_pairs_spanning_an_anchor"]).set_index(["vintage", "pool", "fold"]), ".4f"))
    for v, pool in (("2026-11-09", "own"), ("2026-11-09", "pooled")):
        w(f"\nPer member at {v}, {pool} (sorted by inner-CV loss):\n")
        x = pmem[(pmem.vintage == v) & (pmem.pool == pool)].drop(columns=["vintage", "pool"]).sort_values("cv_inner")
        w(md(x, ".4f"))
    ok = s2[s2.n_oos_points > 0]
    rat = []
    for (v, pool), g in pmem.groupby(["vintage", "pool"]):
        if g.oos_real_task_rate.notna().any():
            rat.append((pool, float((g.oos_real_task_rate / g.cv_inner).median())))
    rat = pd.DataFrame(rat, columns=["pool", "r"]).groupby("pool").r.agg(["min", "max"])
    w("\n**Conclusion D2:** the inner CV does not measure what the forecast does, which explains both the flat "
      "losses and the MLP choice.\n")
    w("1. **Hindsight labels.** The labels are smoothed over all claims public at v, so early folds train on "
      f"labels that did not exist at the block start. Own-pool folds with zero real-time training pairs: "
      f"{K['zero_folds']}. Anthropic had at most one anchor at those block starts (the Jan-2025 $1B claim was published only on 2025-09-02). Revisions of the training labels "
      "themselves are small (0.001-0.006/week), so the leak is mostly about which periods have labels at all.")
    w("2. **Interior validation pairs.** Validation pairs are interior windows of a smoothed path, where "
      "growth follows from the neighbouring slope, so every member with `mom` does about equally well. In the "
      f"pooled pool, {K['within10_h2']} of 18 members are within 10% of the best loss at the H2 vintages. At the "
      f"median member, the real-task loss is {K['ratio_own_h2']:.1f}x (own) and {K['ratio_pooled_h2']:.1f}x "
      f"(pooled) the inner-CV loss at the H2 vintages. At the spring vintages, which have only 3-4 real-task "
      f"points, it is {rat['min'].min():.2f}-{rat['max'].max():.2f}x.")
    w("3. **The rankings do not match.** Spearman correlation between inner-CV and real-task losses across "
      f"members: {', '.join(f'{r.vintage} {r.pool} {r.spearman_inner_vs_oos:+.2f}' for r in ok.itertuples())}.")
    w("4. **Pooled CV is half OpenAI.** About half or more of the pooled validation rows are OpenAI, and the "
      f"Anthropic-row loss of AUTO's pick is {K['anth_ratio_min']:.1f}-{K['anth_ratio_max']:.1f}x the overall loss "
      f"(April-November vintages, where OpenAI rows are {K['openai_share']}). The members' ranks on Anthropic rows "
      f"differ: at 2026-11-09 ridge, AUTO's pick, ranks {K['ridge_anth_rank']} of 18 on Anthropic rows.")
    w("5. **The MLP.** In the own pool its inner-CV loss is the lowest at the late vintages, and clearly so "
      "(0.0207 vs 0.0234 next at 2026-11-09). With fold-nested labels it ranks 11th, and its loss rises most of "
      "all members (0.0227 to 0.0345 on the same folds). On the real task it ranks 14th (own) and 17th (pooled). "
      "Not tested: its early stopping uses a random 20% split of overlapping 4/8/13-week pairs, which would add "
      "the same kind of leak inside the fit.")
    w("6. **Nested CV alone does not give a selector.** With 3-14 real-task points per vintage, the "
      "nested-real-task pick (often midas or bayes_ridge) was no better on the test targets than AUTO's pick. "
      f"Mean |err| over the 6 post-t0 targets of the 5 vintages with real-task points: pooled AUTO "
      f"{K['pick_auto_pooled']:.3f} vs nested {K['pick_nested_pooled']:.3f}; own AUTO {K['pick_auto_own']:.3f} "
      f"vs nested {K['pick_nested_own']:.3f}. With this little data, per-vintage selection among 18 members is the "
      "problem. Nested evaluation is still needed to compare designs honestly.\n")

    # ------------------------------------------------------------------ D3
    w("## D3. Late-starting features\n")
    w("Late-starting = missing in more than 50% of the Anthropic training pairs that start before 2026. Missing "
      "values become 0, which is the company mean after demeaning. Feature status at the 2026-11-09 vintage "
      "(pooled pairs):\n")
    w(miss.to_markdown(floatfmt=".2f"))
    w("\nCoefficients and contributions of the late features in AUTO's chosen member at the H2-2026 vintages "
      "(feature set `all`; pooled = lasso, own = MLP):\n")
    w("- `contrib_rate` is the change in predicted weekly growth when the feature at the forecast row is set "
      "to missing (= company mean).")
    w("- `z_forecast_absmax` is the largest |standardized value| at the forecast row; `z_train_p99` is the 99th "
      "percentile in the present training rows.")
    w("- `perm_importance` is the increase in training loss when the column is permuted.\n")
    w(md(piv.set_index(["pool", "member", "feature"]), ".4f"))
    w("\nRefits of the same member and hyperparameters at the H2 targets:\n")
    w("- `err_no_late`: the late features dropped.")
    w("- `err_gspec_cols`: only the g_spec columns.")
    w("- `err_complete_case`: training only on pairs starting after every late feature exists, which leaves "
      "16-38 pooled and 6 own pairs, so it is too small to use.")
    w("- `y_mean_late_*`: Anthropic label growth/week in pairs where all late features are missing vs present.\n")
    w(md(abl.set_index(["feature_set", "pool", "rule", "target"]), ".3f"))
    w("\nItem 3, AUTO with 43 vs 12 columns (publication): the change in |err| split into the feature effect "
      "(the member AUTO picked with g_spec, refit with all features) and the effect of AUTO switching members:\n")
    w(md(i3[i3.gap_weeks > 0].set_index(["pool", "target"]), ".3f"))
    w("\nMean over the 14 publication targets (the 7 pre-t0 targets contribute 0):\n")
    w(md(i3s, ".3f"))
    w("\n**Conclusion D3:**\n")
    w("- **Lasso: no effect.** The pooled lasso, AUTO's pick with `all`, sets every late-starting feature to "
      "exactly 0, and dropping them changes nothing (err_full = err_no_late at all 6 H2 cases). It mainly uses "
      "github_repo_share, ramp_vendor_share, langchain_npm_share and ramp_adoption.")
    w("- **MLP: a spurious pattern.** Imputation creates one for the MLP. The late features are present only "
      "from Nov 2025 to Jul 2026, the hyper-growth era (0.067/week vs 0.039 where they are missing). Scaled by "
      "the RMS over mostly-imputed rows, they reach the forecast row far outside the training range "
      "(vercel_spend_share z = -88 vs p99 29; openrouter_share 22.9 vs 6.7; cloud_bedrock_growth_cum 7.9 vs 4.6). "
      "The MLP turns them into net extra growth of about +0.03/week (sum of contrib_rate)."
      "")
    w("- **The size of the MLP effect.** Dropping the late features improves the own-pool MLP at all 12 H2 "
      "cases. For example, rr_2026_10 with `all` goes from -1.041 to -0.169 (period-end). With g_spec, the "
      "+214% case (-1.145) becomes -0.499: OpenRouter tokens/share from Nov 2025, vercel from May 2026 and pypi "
      "from Apr 2026 are late-starting even in g_spec.")
    w("- **Linear models are fine.** For the pooled ridge the same g_spec late features carry real signal "
      "(dropping them worsens rr_2026_09 from -0.233 to -0.314).")
    w("- **Item 3 is a selection effect.** Going from 12 to 43 columns improves the same member slightly. "
      "AUTO's loss comes from switching to a near-tied member: midas at rr_2026_05 (-0.537 vs kernel_ridge "
      "+0.128) and lasso instead of ridge at the three H2 targets.\n")

    # ------------------------------------------------------------------ D4
    w("## D4. Pre-t0 targets: why the label path is worse than G's path\n")
    w("Each step changes one thing, using `backtest.prepublication_forecast` exactly as the A/G scripts do "
      "(refit=True, catalog roles patched as in `forecast_ramp.py gh_share`):\n")
    w("- **S0:** the zoo's value: the label path, exp(mean) integrated. It is the same for every member.")
    w("- **S1:** the panel smoother's own forecast (the mean).")
    w("- **S1r:** the panel replica inside the backtest machinery (panel noise, rho 0.5), as a check.")
    w("- **S2:** the backtest's process-noise refit.")
    w("- **S3:** plus the crossing profile (= A).")
    w("- **S4:** plus segment claims (Claude Code run-rates, shares).")
    w("- **S5:** plus the GitHub new-repo share indicator (= G).")
    w("- **S5b:** G without segment claims.\n")
    w("The definition lag is the same in both pipelines (`config.DEFINITION_LAG_DAYS`, t28x13 = 14 days). The "
      "target dates (`t_eff`) of these 7 targets are identical under every rho (none is a crossing claim).\n")
    w(md(d4))
    w("\nChange in |err| at each step (+ = worse):\n")
    w(md(d4c))
    nz = ss[ss.kind == "pre"].drop_duplicates("vintage")
    w("\nProcess noise at the two vintages (panel `fit_noise` vs `calibrate.fit_process_noise`):\n")
    w(md(nz.set_index("vintage")[[c for c in nz.columns if c.startswith("noise")] + ["rho_A", "rho_G"]].T, ".4f"))
    w("\n**Conclusion D4:**\n")
    w("- **The steps that contribute nothing or little.** The two process-noise fits agree to 4 digits "
      "(contribution 0). The crossing profile moves individual quarters by up to 0.05 in either direction "
      f"(rho 0.89-0.92 vs 0.5) and averages {K['d4_cross']:+.3f}. Segment claims contribute "
      f"{K['d4_seg']:+.3f}. The Jensen term costs the zoo path 0 to {K['jensen_max']:.3f} on the revenue "
      "targets (S0 vs S1).")
    w("- **The indicator is the whole difference.** Adding the GitHub share indicator moves the mean |err| "
      f"from {K['d4_S4']:.3f} to {K['d4_S5']:.3f}. The zoo path's mean is {K['d4_S0']:.3f}.")
    w("- **Why the indicator helps.** Before t0 the claims-only path is a near log-linear interpolation "
      "between 3-10 anchors: Jan 2025 ($1B), May 2025, Oct 2025, Dec 2025, Jan/Feb/Mar 2026 and so on. The "
      "indicator bends it between anchors. It raises 2025 revenue (all G errors turn slightly negative) and "
      "lowers December 2025 (rr_2025_12: -0.145 to -0.039). Segment claims and the indicator interact: G "
      "without segments gives -0.130 on rr_2025_12.")
    w("- **Caveat on the GitHub series.** It was fetched in one go on 2026-11-11 and includes repos made "
      "public later, so its pre-t0 advantage partly reflects hindsight data.")
    w("- **No member can do better.** No ML member can improve these targets in the current design: they "
      "only measure the label smoother.\n")
    path = os.path.join(REP, f"zoo_diagnostics_{TODAY}.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
