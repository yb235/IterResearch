# Run from the repo root:
#   OMP_NUM_THREADS=1 PYTHONPATH=. python runrate_nowcast/scripts/register_q4.py [--dry-run | --register] [--jobs 4]
"""Frozen forward forecasts at vintage 2026-11-12, filings excluded (DATA_PLAN step 4, sub-task H).

The backtests so far scored models on numbers that were known when the models were chosen, so
their scores are optimistic. This script makes forecasts of numbers nobody knows yet, so they
can be registered now and scored when the numbers are published:

  revenue_q4_2026     Anthropic revenue 2026-10-01..2026-12-31, $B (GAAP basis)
  runrate_2026-11-30  official-basis run-rate (t28x13 definition) on 2026-11-30, $B
  runrate_2026-12-31  the same on 2026-12-31

MODELS (frozen 2026-11-12, before any of their values was seen; never add or drop one)
  A   claims only: state-space, every parameter refit on data public at the vintage, crossing
      position profiled (backtest.prepublication_forecast(refit=True, crossing="profile"))
  B   claims + the catalog's default indicators (npm / PyPI SDK downloads, Ramp token spend)
  E   claims + OpenRouter Anthropic tokens x measured price per token (forecast_openrouter.py or_price)
  G   claims + share of new GitHub repos that mention Claude (forecast_ramp.py gh_share)
  R   claims + Ramp business adoption of Anthropic (forecast_ramp.py ramp_only)
  P1  panel PCA + ridge, pooled: the pre-declared primary panel model (panel_model.forecast_target)
  P2  panel ridge, pooled
  N   naive: the last non-filing point run-rate claim carried forward (backtest._naive on A's evidence)
  C   equal-weight combination: exp(mean of the log point forecasts of A, E, G, R and P2). Defined
      only when all five members give the quantity. Point only.
Point forecast: the median for a state-space run-rate, the mean for state-space revenue, and the
model's single number for P1, P2, N and C (which give no interval).

How each forecast is made
  * Each quantity becomes a target row added to the model's table (include = 0, published
    2099-12-31, an origin of its own). backtest.prepublication_forecast then forecasts it exactly
    as it forecasts a backtest target, at vintage 2026-11-12: evidence = rows published on or
    before the vintage, filing rows excluded (config.USE_FILINGS is False; backtest.check_no_leak
    and assert_clean_evidence below both check it), parameters refit on data public at the vintage.
  * Each state-space variant's table and catalog roles are built as its backtest script builds them
    (variant_data). The catalog patch stays in place for the whole forecast, because the refit
    restarts the indicator priors from the catalog.
  * Registered quantiles are the model's own uncertainty about the quantity.
    prepublication_forecast adds the target tier's measurement noise to the sd (official run-rate:
    0.04 in logs; GAAP revenue: 0.3%). That noise is removed here, because registry.score adds the
    noise of whichever claim it scores against. registry.register uses the same convention.
  * Panel models: panel_model.forecast_target on the same target rows. Labels and process noise are
    refit at the vintage, and filing rows are dropped when the claims are loaded.

Outputs
  --dry-run (default)  print the table and write reports/q4_forecasts_2026-11-12.csv. The registry
                       is not touched.
  --register           the same, then registry.register_summary once per model (q4_frozen_<letter>).
                       Refuses if code or data under runrate_nowcast/ have uncommitted changes (the
                       registry pins the git commit) or if q4_frozen_ entries are already registered.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import math
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

from runrate_nowcast import altdata, backtest, calibrate, config, data, pricing, registry
from runrate_nowcast import features as F
from runrate_nowcast import panel_model as pm
from runrate_nowcast.model import Model, Z80, lognormal_summary, period_key

VINTAGE = pd.Timestamp("2026-11-12")
Q4_START, Q4_END = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-12-31")
Q4_PERIOD = period_key(Q4_START, Q4_END)
RUNRATE_DATES = ("2026-11-30", "2026-12-31")
REVENUE = "revenue_q4_2026"
QUANTITIES = (REVENUE,) + tuple(f"runrate_{d}" for d in RUNRATE_DATES)
TARGET_IDS = {q: f"q4_frozen_target_{q}" for q in QUANTITIES}
TARGET_PUBLISHED = pd.Timestamp("2099-12-31")   # placeholder: the vintage bounds the evidence, not this
TARGET_VALUE = 1.0                              # placeholder: the values are unknown
NOTE = "frozen 2026-11-12, filings excluded"
POINT_ONLY_NOTE = "; point forecast only, no interval (q10 = q90 = point)"
CSV_PATH = os.path.join(config.OUTPUT_DIR, "q4_forecasts_2026-11-12.csv")
REGISTRY_PREFIX = "q4_frozen_"

# Frozen 2026-11-12. Do not add or drop a model after seeing their values.
MODELS = {
    "A": "claims only (state-space, refit at the vintage, crossing position profiled)",
    "B": "claims + SDK downloads (catalog default indicators)",
    "E": "claims + OpenRouter Anthropic tokens x price per token (or_price)",
    "G": "claims + GitHub new-repo share (gh_share)",
    "R": "claims + Ramp business adoption (ramp_only)",
    "P1": "panel PCA + ridge, pooled (pre-declared primary panel model)",
    "P2": "panel ridge, pooled",
    "N": "naive: last non-filing point claim carried forward",
    "C": "equal-weight mean of the log point forecasts of A, E, G, R, P2",
}
STATE_SPACE = ("A", "B", "E", "G", "R")
PANEL = {"P1": "pca", "P2": "ridge"}           # model letter -> panel_model kind (pool: pooled)
NAIVE = "N"
COMBINED = "C"
COMBINATION = ("A", "E", "G", "R", "P2")
# Indicators given catalog role "model" in each state-space variant; every other "model" row becomes
# "monitor". None = the catalog as stored. With the current catalog this reproduces the patches in
# forecast_openrouter.py (or_price) and forecast_ramp.py (gh_share, ramp_only).
VARIANT_INDICATORS = {"A": None, "B": None, "E": ("openrouter_anthropic_tokens",),
                      "G": ("github_repo_share_claude",), "R": ("ramp_adoption_anthropic",)}
JOBS = ("B", "E", "G", "R", "A", "panel")      # slowest first; job "A" also computes N
COLS = ["model", "registry_name", "quantity", "stat", "point", "q10", "q90", "log_mu", "log_sd",
        "sd_usd_b", "n_evidence", "detail"]


# --------------------------------------------------------------------------- target rows
def target_rows() -> pd.DataFrame:
    """One row per quantity in the observations.csv schema. include = 0 and a far-future
    publication date, so data.select never returns them as evidence."""
    common = dict(kind="point", value=TARGET_VALUE, date_sd_days=0.0, published=TARGET_PUBLISHED,
                  source="register_q4.py", url="", include=0, notes="forward target, value unknown",
                  segment="total", indicator="", is_forecast=0, filing=0, date_lo=pd.NaT, date_hi=pd.NaT,
                  date_kind="")
    rows = [dict(common, obs_id=TARGET_IDS[REVENUE], metric="revenue", ref_date=Q4_END,
                 period_start=Q4_START, period_end=Q4_END, definition="", tier="gaap")]
    for d in RUNRATE_DATES:
        rows.append(dict(common, obs_id=TARGET_IDS[f"runrate_{d}"], metric="run_rate", ref_date=pd.Timestamp(d),
                         period_start=pd.NaT, period_end=pd.NaT, definition="t28x13", tier="official"))
    t = pd.DataFrame(rows)
    t["origin_id"] = t["obs_id"]
    return t


def with_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([df, target_rows()], ignore_index=True)
    if out.obs_id.duplicated().any():
        raise ValueError("target obs_id already in the table")
    return out


def target_row(tbl: pd.DataFrame, quantity: str):
    """The target as prepublication_forecast reads it (with t_eff), as in its crossing branch."""
    return tbl[tbl.obs_id == TARGET_IDS[quantity]].assign(t_eff=lambda x: x.apply(data.effective_date, axis=1)).iloc[0]


def filing_obs_ids() -> set:
    raw = data.load()
    return set(raw.loc[raw.filing.fillna(0).astype(int) != 0, "obs_id"])


def assert_clean_evidence(ev: pd.DataFrame, vintage=VINTAGE, filing_ids: set | None = None) -> None:
    """Evidence must be public at the vintage, hold no filing row (by flag and by obs_id in
    observations.csv) and none of the forward targets."""
    v = pd.Timestamp(vintage)
    if "filing" in ev and (ev.filing.fillna(0).astype(int) != 0).any():
        raise AssertionError(f"filing rows in evidence: {ev[ev.filing.fillna(0) != 0].obs_id.tolist()}")
    hit = set(ev.obs_id) & (filing_obs_ids() if filing_ids is None else filing_ids)
    if hit:
        raise AssertionError(f"filing rows in evidence: {sorted(hit)}")
    if (pd.to_datetime(ev.published) > v).any():
        raise AssertionError(f"evidence published after {v:%Y-%m-%d}")
    if ev.obs_id.isin(list(TARGET_IDS.values())).any() or ev.origin_id.isin(list(TARGET_IDS.values())).any():
        raise AssertionError("a forward target is in the evidence")


# --------------------------------------------------------------------------- records
def _nan_record(quantity: str, stat: str, **extra) -> dict:
    return {"quantity": quantity, "stat": stat, "point": np.nan, "q10": np.nan, "q90": np.nan,
            "log_mu": np.nan, "log_sd": np.nan, "sd_usd_b": np.nan, "n_evidence": np.nan, "detail": "", **extra}


def point_record(quantity: str, value: float, detail: str = "", **extra) -> dict:
    """A point forecast without an interval."""
    r = _nan_record(quantity, "point", detail=detail, **extra)
    if value is not None and np.isfinite(value) and value > 0:
        r.update(point=float(value), log_mu=math.log(value))
    return r


def revenue_quantiles(mean: float, sd: float) -> tuple[float, float]:
    """q10 and q90 of a period revenue: model.Result.period_revenue's lognormal moment match."""
    if mean > 0:
        s2 = math.log1p((sd / mean) ** 2)
        lm = math.log(mean) - s2 / 2
        return math.exp(lm - Z80 * math.sqrt(s2)), math.exp(lm + Z80 * math.sqrt(s2))
    return mean - Z80 * sd, mean + Z80 * sd


def latent(row, mu: float, sd: float, tiers: dict) -> tuple[float, float]:
    """Remove the target tier's measurement noise (and bias) that observation_predictive adds,
    leaving the model's own mean and sd of the quantity."""
    tier = tiers[row.tier]
    if row.metric == "revenue":
        return mu, math.sqrt(max(sd ** 2 - (tier["log_sd"] * mu) ** 2, 0.0))   # noise is scaled by the forecast mean
    # run-rate: base_var = log_sd^2; the date term is 0 because the targets have date_sd_days = 0
    return mu - tier.get("bias", 0.0), math.sqrt(max(sd ** 2 - tier["log_sd"] ** 2, 0.0))


def statespace_record(quantity: str, row, mu: float, sd: float, tiers: dict, n_evidence: int) -> dict:
    m, s = latent(row, mu, sd, tiers)
    if row.metric == "revenue":
        q10, q90 = revenue_quantiles(m, s)
        return {"quantity": quantity, "stat": "mean", "point": m, "q10": q10, "q90": q90,
                "log_mu": math.log(m) if m > 0 else np.nan, "log_sd": np.nan, "sd_usd_b": s,
                "n_evidence": n_evidence, "detail": ""}
    ln = lognormal_summary(m, s)
    return {"quantity": quantity, "stat": "median", "point": ln["median"], "q10": ln["q10"], "q90": ln["q90"],
            "log_mu": m, "log_sd": s, "sd_usd_b": np.nan, "n_evidence": n_evidence, "detail": ""}


# --------------------------------------------------------------------------- state-space variants
def variant_catalog_table(model: str, cat: pd.DataFrame) -> pd.DataFrame:
    use = VARIANT_INDICATORS[model]
    if use is None:
        return cat
    c = cat.copy()
    c.loc[c.role == "model", "role"] = "monitor"
    c.loc[c.indicator.isin(use), "role"] = "model"
    return c


@contextlib.contextmanager
def variant_catalog(model: str):
    """Inside the block, altdata.catalog() returns the variant's roles. data.load_all,
    calibrate.load_params, backtest.unfitted_params and the refit all read the catalog through it."""
    if VARIANT_INDICATORS[model] is None:
        yield
        return
    orig = altdata.catalog
    cat0 = orig()

    def patched(path=config.CATALOG_PATH):
        return variant_catalog_table(model, cat0)

    altdata.catalog = patched
    try:
        yield
    finally:
        altdata.catalog = orig


def variant_data(model: str) -> tuple[pd.DataFrame, dict]:
    """Table and parameters of a state-space model, built as its backtest script builds them.
    Call inside variant_catalog(model)."""
    df = data.load_all()
    if model == "A":                 # forecast_statespace_ml.py, claims_only
        base = df[df.metric != "indicator"]
        df = base[(base.segment == "total") & (base.metric != "share")]
    elif model == "E":               # forecast_openrouter.py or_price
        df = pricing.adjust_indicators(df, 1.0, "github")
    return df, calibrate.load_params()


def statespace_forecasts(model: str, df: pd.DataFrame | None = None, params: dict | None = None,
                         refit: bool = True, crossing="profile", vintage=VINTAGE,
                         quantities=QUANTITIES) -> tuple[dict, dict]:
    """({quantity: record}, {quantity: evidence}) for one state-space model.

    df / params: the model's table and parameters (default: variant_data). refit and crossing are
    the frozen settings; tests pass refit=False, crossing=None for speed.
    """
    filing_ids = filing_obs_ids()
    with variant_catalog(model):
        if df is None:
            df, p0 = variant_data(model)
            params = p0 if params is None else params
        elif params is None:
            params = calibrate.load_params()
        tbl = with_targets(df)
        tiers = Model(params).p["tiers"]
        recs, evs = {}, {}
        for q in quantities:
            row = target_row(tbl, q)
            f = backtest.prepublication_forecast(tbl, row, params, refit=refit, crossing=crossing, vintage=vintage)
            if f is None:
                recs[q] = _nan_record(q, "median" if row.metric == "run_rate" else "mean", detail="too little evidence")
                continue
            mu, sd, n_ev, ev = f
            assert_clean_evidence(ev, vintage, filing_ids)
            recs[q], evs[q] = statespace_record(q, row, mu, sd, tiers, n_ev), ev
    return recs, evs


def naive_forecasts(evidence: dict) -> dict:
    """N: backtest._naive on A's evidence (the last point run-rate claim, held flat)."""
    rows = target_rows().set_index("obs_id", drop=False)
    out = {}
    for q in QUANTITIES:
        ev = evidence.get(q)
        if ev is None:
            out[q] = point_record(q, np.nan, "no evidence")
            continue
        row = rows.loc[TARGET_IDS[q]]
        nv = backtest._naive(row, ev)
        val = np.nan if nv is None else (math.exp(nv) if row.metric == "run_rate" else nv)
        rr = ev[(ev.metric == "run_rate") & (ev.kind == "point") & (ev.segment == "total") & (ev.is_forecast == 0)]
        last = rr.sort_values("ref_date").iloc[-1] if len(rr) else None
        detail = "" if last is None else f"last point claim {last.obs_id} = {last.value:g} @ {last.ref_date:%Y-%m-%d}"
        out[q] = point_record(q, val, detail, n_evidence=len(ev))
    return out


# --------------------------------------------------------------------------- panel
def panel_claims() -> dict:
    """Claims tables of the panel (panel_model.load_claims drops filing rows at load)."""
    return {c: pm.load_claims(c) for c in pm.COMPANIES}


def panel_target(quantity: str):
    r = target_rows().set_index("obs_id", drop=False).loc[TARGET_IDS[quantity]]
    return pm._Row(company="anthropic", t_eff=data.effective_date(r), **r.to_dict())


def panel_forecasts(claims: dict | None = None, sources: dict | None = None, noise: dict | None = None,
                    vintage=VINTAGE, quantities=QUANTITIES) -> dict:
    """{P1: {quantity: record}, P2: {...}} from panel_model.forecast_target (pooled pool only).
    noise: {company: process noise} to skip the refit (tests only)."""
    claims = panel_claims() if claims is None else claims
    sources = F.load_sources() if sources is None else sources
    out = {m: {} for m in PANEL}
    for q in quantities:
        rows = pm.forecast_target(panel_target(q), vintage, claims, sources, models=tuple(PANEL.values()),
                                  pools=("pooled",), params=noise)
        by = {r["model"]: r for r in rows}
        for m, kind in PANEL.items():
            r = by.get(f"panel {kind} (pooled)")
            if r is None:
                out[m][q] = point_record(q, np.nan, "panel gave no forecast (too few anchors or pairs)")
                continue
            detail = (f"t0 {r['t0']}, gap {r['gap_weeks']:.1f} wk, {r['n_train_pairs']} pairs, "
                      f"{r['hyper']}, {r['cv_folds']} CV folds")
            out[m][q] = point_record(q, r["forecast"], detail, n_evidence=r["n_anchors"])
    return out


# --------------------------------------------------------------------------- combination
def combine(results: dict, members=COMBINATION) -> dict:
    """C: exp(mean of the members' log point forecasts) per quantity. NaN unless every member
    gives the quantity (the rule is fixed now, so a missing member cannot be dropped later)."""
    out = {}
    for q in QUANTITIES:
        pts = np.array([results.get(m, {}).get(q, {}).get("point", np.nan) for m in members], dtype=float)
        ok = bool(np.all(np.isfinite(pts) & (pts > 0)))
        val = float(np.exp(np.mean(np.log(pts)))) if ok else np.nan
        out[q] = point_record(q, val, "members " + ", ".join(members) + ("" if ok else " (a member is missing)"))
    return out


# --------------------------------------------------------------------------- running
def run_job(name: str) -> tuple[str, dict, float]:
    warnings.simplefilter("ignore")
    t = time.time()
    if name == "panel":
        res = panel_forecasts()
    elif name == "A":
        recs, evs = statespace_forecasts("A")
        res = {"A": recs, NAIVE: naive_forecasts(evs)}
    else:
        res = {name: statespace_forecasts(name)[0]}
    return name, res, time.time() - t


def compute_all(jobs: int = 1, verbose: bool = True) -> tuple[dict, dict]:
    """({model: {quantity: record}} for every model in MODELS, {job: seconds})."""
    results, timings = {}, {}

    def done(name, res, sec):
        results.update(res)
        timings[name] = sec
        if verbose:
            print(f"  {name}: {sec:.0f}s", flush=True)

    if jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("fork")) as ex:
            for fut in as_completed([ex.submit(run_job, j) for j in JOBS]):
                done(*fut.result())
    else:
        for j in JOBS:
            done(*run_job(j))
    results[COMBINED] = combine(results)
    return {m: results[m] for m in MODELS}, timings


# --------------------------------------------------------------------------- output
def _sha256(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def inputs_sha256() -> str:
    """One hash over every file in data/ plus params.json (each file's path and sha256)."""
    files = sorted(p for p in glob.glob(os.path.join(config.PKG_DIR, "data", "**", "*"), recursive=True)
                   if os.path.isfile(p)) + [config.PARAMS_PATH]
    h = hashlib.sha256()
    for p in files:
        h.update(f"{os.path.relpath(p, config.PKG_DIR)} {_sha256(p)}\n".encode())
    return h.hexdigest()


def uncommitted() -> list[str]:
    """Uncommitted changes (incl. untracked files) under runrate_nowcast/, outputs excluded."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "."],
                         cwd=config.PKG_DIR, capture_output=True, text=True)
    if out.returncode != 0:
        return [f"git status failed: {out.stderr.strip()}"]
    return [ln for ln in out.stdout.splitlines()
            if ln.strip() and "runrate_nowcast/reports/" not in ln and "runrate_nowcast/forecasts/" not in ln]


def metadata() -> dict:
    return {"vintage": f"{VINTAGE:%Y-%m-%d}", "git_commit": registry._git_commit(),
            "uncommitted_changes": len(uncommitted()), "data_sha256": _sha256(config.DATA_PATH),
            "indicators_sha256": _sha256(config.INDICATORS_PATH), "inputs_sha256": inputs_sha256()}


def to_frame(results: dict, meta: dict | None = None) -> pd.DataFrame:
    rows = []
    for m, desc in MODELS.items():
        for q in QUANTITIES:
            r = dict(results[m][q])
            r.update(model=m, registry_name=REGISTRY_PREFIX + m, description=desc)
            rows.append(r)
    out = pd.DataFrame(rows)[COLS + ["description"]]
    for k, v in (meta or {}).items():
        out[k] = v
    return out


def table(frame: pd.DataFrame) -> str:
    def cell(r, nd):
        if not np.isfinite(r.point):
            return "n/a"
        s = f"{r.point:.{nd}f}"
        return s + (f" [{r.q10:.{nd}f}, {r.q90:.{nd}f}]" if np.isfinite(r.q10) else "")

    heads = {REVENUE: "Q4 2026 revenue (mean [q10, q90])"} | {
        f"runrate_{d}": f"Run-rate {d} (median [q10, q90])" for d in RUNRATE_DATES}
    lines = ["| Model | " + " | ".join(heads[q] for q in QUANTITIES) + " |", "|---" * (len(QUANTITIES) + 1) + "|"]
    for m in MODELS:
        f = frame[frame.model == m].set_index("quantity")
        lines.append(f"| {m} {MODELS[m]} | " + " | ".join(cell(f.loc[q], 2 if q == REVENUE else 1)
                                                          for q in QUANTITIES) + " |")
    return "\n".join(lines)


def registry_summary(recs: dict) -> tuple[dict, bool]:
    """register_summary's format: {"runrate_<date>": {median, q10, q90},
    "revenue_quarter": {mean, q10, q90}, "period": "<key>"}. A point-only model gets q10 = q90 =
    point (register_summary needs both; the registered sd is then 0). Returns (summary, point_only)."""
    s, point_only = {}, False

    def bounds(r):
        nonlocal point_only
        if np.isfinite(r["q10"]) and np.isfinite(r["q90"]):
            return float(r["q10"]), float(r["q90"])
        point_only = True
        return float(r["point"]), float(r["point"])

    for d in RUNRATE_DATES:
        r = recs[f"runrate_{d}"]
        if np.isfinite(r["point"]):
            lo, hi = bounds(r)
            s[f"runrate_{d}"] = {"median": float(r["point"]), "q10": lo, "q90": hi}
    r = recs[REVENUE]
    if np.isfinite(r["point"]):
        lo, hi = bounds(r)
        s["revenue_quarter"] = {"mean": float(r["point"]), "q10": lo, "q90": hi}
        s["period"] = Q4_PERIOD
    return s, point_only


def register_all(results: dict, path: str = config.REGISTRY_PATH) -> list[dict]:
    entries = []
    for m in MODELS:
        summ, point_only = registry_summary(results[m])
        if not summ:
            print(f"  {REGISTRY_PREFIX}{m}: nothing to register (no quantity)")
            continue
        entries.append(registry.register_summary(REGISTRY_PREFIX + m, VINTAGE, summ,
                                                 note=NOTE + (POINT_ONLY_NOTE if point_only else ""), path=path))
    return entries


def already_registered(path: str = config.REGISTRY_PATH) -> list[str]:
    return [e["model"] for e in registry.load_registry(path) if str(e.get("model", "")).startswith(REGISTRY_PREFIX)]


def _compare_with_previous(frame: pd.DataFrame, path: str) -> None:
    if not os.path.exists(path):
        return
    old = pd.read_csv(path).set_index(["model", "quantity"])[["point", "q10", "q90"]]
    new = frame.set_index(["model", "quantity"])[["point", "q10", "q90"]]
    j = new.join(old, rsuffix="_csv", how="outer")
    d = max(float(np.nanmax(np.abs(j[c] / j[f"{c}_csv"] - 1).to_numpy(float), initial=0.0)) for c in ("point", "q10", "q90"))
    print(f"Largest relative difference from the existing {os.path.basename(path)}: {d:.2e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Frozen Q4 2026 forecasts at vintage 2026-11-12 (filings excluded).")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default: print the table and write the CSV only")
    mode.add_argument("--register", action="store_true", help="also append one registry entry per model")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="parallel processes")
    ap.add_argument("--out", default=CSV_PATH, help="CSV path")
    ap.add_argument("--allow-dirty", action="store_true", help="register despite uncommitted changes")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    if config.USE_FILINGS:
        raise SystemExit("config.USE_FILINGS must be False: the filings are the answers, not inputs")
    if a.register:
        prev = already_registered()
        if prev:
            raise SystemExit(f"already registered: {prev}. The first registration is the record.")
        dirty = uncommitted()
        if dirty and not a.allow_dirty:
            raise SystemExit("uncommitted changes under runrate_nowcast/ (commit first, so the registered git "
                             "commit pins this code and data):\n  " + "\n  ".join(dirty))
    t0 = time.time()
    print(f"Forecasts at vintage {VINTAGE:%Y-%m-%d}, filings excluded, {a.jobs} job(s):", flush=True)
    results, _ = compute_all(a.jobs)
    frame = to_frame(results, metadata())
    print()
    print(table(frame))
    print(f"\nRuntime {time.time() - t0:.0f}s. Intervals are 80% (q10, q90), the model's own uncertainty; "
          "P1, P2, N and C are point forecasts.")
    if a.register:
        _compare_with_previous(frame, a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    frame.to_csv(a.out, index=False)
    print(f"-> {a.out}")
    if not a.register:
        print("Dry run: nothing registered.")
        return 0
    entries = register_all(results)
    for e in entries:
        print(f"registered {e['model']} at {e['registered_at_utc']} (commit {e['git_commit']})")
    print(f"-> {config.REGISTRY_PATH} (commit it now)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
