# Run from the repo root:
#   OMP_NUM_THREADS=1 PYTHONPATH=. python runrate_nowcast/scripts/register_q4_v2.py [--dry-run | --register] [--jobs 4]
"""Sub-task N: re-register the Q4 2026 forecasts with the primary lower-claim rule (iv115), as of 2026-11-13.

DATA_PLAN.md 子任务 N (commit abce18a) registered, before any rerun, that "surpassed / over $X" claims
(kind = lower) mean X <= true value <= 1.15 X (lower_rule.py). The q4_frozen_ entries of 2026-11-12
(scripts/register_q4.py) used the one-sided rule; they stay in the registry untouched. This script
adds two new sets, same nine models and the same frozen specifications as register_q4.py (A, B, E, G,
R state-space, P1 / P2 panel pooled, N naive, C equal-weight combination of A, E, G, R, P2), the same
three quantities and target rows, the same latent-noise convention and registry format:

  q4_v2_<model>     filings excluded, as every earlier registration (config.USE_FILINGS False;
                    backtest.check_no_leak and register_q4.assert_clean_evidence on every evidence set)
  q4_v2pub_<model>  also uses the filing rows (filing in {1, 2}: S-1, S-1/A, and third-party numbers
                    that may derive from them) published on or before 2026-11-13 as evidence, e.g. the
                    S-1/A $81B run-rate at 2026-10-31, ~$71B at 2026-09-30 and Q1-Q3 2026 GAAP revenue.
                    config.USE_FILINGS is True inside these jobs only; the panel's claims tables keep
                    their filing rows and pm.public_claims is replaced, in these jobs only, by the same
                    selection without the filing filter. Evidence must still be public by the vintage.

Each registry entry is register_summary's entry plus: set, lower_claim_rule, filings_as_evidence,
inputs_sha256 (every data file + params.json), openai_claims_sha256, code_sha256 of the files that
define the rule and the forecasts, and the list of uncommitted changes at registration (the code of
sub-task N is registered before the orchestrator commits it; code_sha256 pins it).

Outputs
  --dry-run (default)  print both tables (and write --out if given). The registry is not touched.
  --register           the same, write reports/q4_forecasts_v2_2026-11-13.csv, then append one entry per
                       model and set to forecasts/registry.jsonl (append only; refuses if q4_v2 entries exist)
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

from runrate_nowcast import backtest, config, data, lower_rule, registry
from runrate_nowcast import panel_model as pm
from runrate_nowcast.model import Model

_spec = importlib.util.spec_from_file_location("register_q4", os.path.join(os.path.dirname(__file__), "register_q4.py"))
rq = importlib.util.module_from_spec(_spec)
sys.modules["register_q4"] = rq
_spec.loader.exec_module(rq)

VINTAGE = pd.Timestamp("2026-11-13")
VARIANT = lower_rule.PRIMARY                       # iv115
SETS = {"q4_v2_": False, "q4_v2pub_": True}       # prefix -> filings as evidence
CSV_PATH = os.path.join(config.OUTPUT_DIR, "q4_forecasts_v2_2026-11-13.csv")
NOTES = {False: "v2 2026-11-13: lower claims as [X, 1.15X] (iv115, DATA_PLAN 子任务 N), filings excluded",
         True: "v2pub 2026-11-13: lower claims as [X, 1.15X] (iv115), + filing rows (S-1, S-1/A, filing 1/2) "
               "published by 2026-11-13 as evidence"}
CODE_FILES = ("model.py", "config.py", "lower_rule.py", "data.py", "backtest.py", "calibrate.py", "robust.py",
              "panel_model.py", "altdata.py", "pricing.py", "registry.py", "scripts/register_q4.py",
              "scripts/register_q4_v2.py")
JOBS = [(f, j) for j in ("B", "E", "G", "R", "A", "panel") for f in (False, True)]     # slowest first


# --------------------------------------------------------------------------- filings as evidence
def claims_with_filings(company: str) -> pd.DataFrame:
    """pm.load_claims without dropping filing rows."""
    df = data.load(pm.CLAIMS_PATHS[company])
    df = df[(df.segment == "total") & df.metric.isin(["run_rate", "revenue"])].copy()
    df["company"] = company
    return df.reset_index(drop=True)


def public_claims_with_filings(claims, vintage, exclude_origin=None, start=None):
    """pm.public_claims without the filing filter (published <= vintage still required)."""
    v = pd.Timestamp(vintage)
    raw = claims[(claims.published <= v) & (claims.include == 1) & (claims.is_forecast == 0)]
    if exclude_origin is not None:
        raw = raw[raw.origin_id != exclude_origin]
    sel = data.select(raw, vintage=v, forecasts=False, filings=True)
    if start is not None and len(sel):
        s = pd.Timestamp(start)
        sel = sel[~((sel.metric == "revenue") & (pd.to_datetime(sel.period_start) < s))].reset_index(drop=True)
    if len(sel) and (pd.to_datetime(sel.published) > v).any():
        raise AssertionError("claims published after the vintage")
    if exclude_origin is not None and (sel.origin_id == exclude_origin).any():
        raise AssertionError("target origin in claims")
    return sel


@contextlib.contextmanager
def filings_as_evidence(on: bool):
    if not on:
        yield
        return
    old, old_pc = config.USE_FILINGS, pm.public_claims
    config.USE_FILINGS = True
    pm.public_claims = public_claims_with_filings
    try:
        yield
    finally:
        config.USE_FILINGS, pm.public_claims = old, old_pc


def assert_evidence(ev: pd.DataFrame, filings: bool, filing_ids: set) -> None:
    if not filings:
        rq.assert_clean_evidence(ev, VINTAGE, filing_ids)
        return
    if (pd.to_datetime(ev.published) > VINTAGE).any():
        raise AssertionError(f"evidence published after {VINTAGE:%Y-%m-%d}")
    if ev.obs_id.isin(list(rq.TARGET_IDS.values())).any() or ev.origin_id.isin(list(rq.TARGET_IDS.values())).any():
        raise AssertionError("a forward target is in the evidence")


# --------------------------------------------------------------------------- forecasts
def statespace(model: str, filings: bool) -> tuple[dict, dict]:
    """register_q4.statespace_forecasts at VINTAGE (same table, params, refit, crossing profile)."""
    filing_ids = rq.filing_obs_ids()
    with rq.variant_catalog(model):
        df, params = rq.variant_data(model)
        tbl = rq.with_targets(df)
        tiers = Model(params).p["tiers"]
        recs, evs = {}, {}
        for q in rq.QUANTITIES:
            row = rq.target_row(tbl, q)
            f = backtest.prepublication_forecast(tbl, row, params, refit=True, crossing="profile", vintage=VINTAGE)
            if f is None:
                recs[q] = rq._nan_record(q, "median" if row.metric == "run_rate" else "mean", detail="too little evidence")
                continue
            mu, sd, n_ev, ev = f
            assert_evidence(ev, filings, filing_ids)
            recs[q], evs[q] = rq.statespace_record(q, row, mu, sd, tiers, n_ev), ev
            if filings:
                recs[q]["detail"] = f"{int((ev.filing.fillna(0).astype(int) != 0).sum())} filing rows in evidence"
    return recs, evs


def run_job(job) -> tuple[tuple, dict, float]:
    filings, name = job
    warnings.simplefilter("ignore")
    lower_rule.apply(VARIANT)
    t = time.time()
    with filings_as_evidence(filings):
        if name == "panel":
            claims = {c: (claims_with_filings(c) if filings else pm.load_claims(c)) for c in pm.COMPANIES}
            res = rq.panel_forecasts(claims=claims, vintage=VINTAGE)
        elif name == "A":
            recs, evs = statespace("A", filings)
            res = {"A": recs, rq.NAIVE: rq.naive_forecasts(evs)}
        else:
            res = {name: statespace(name, filings)[0]}
    return job, res, time.time() - t


def compute_all(jobs: int) -> dict:
    """{filings: {model: {quantity: record}}}"""
    out = {False: {}, True: {}}
    if jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("fork")) as ex:
            for fut in as_completed([ex.submit(run_job, j) for j in JOBS]):
                job, res, sec = fut.result()
                out[job[0]].update(res)
                print(f"  {'pub' if job[0] else 'nofile'} {job[1]}: {sec:.0f}s", flush=True)
    else:
        for j in JOBS:
            job, res, sec = run_job(j)
            out[job[0]].update(res)
            print(f"  {'pub' if job[0] else 'nofile'} {job[1]}: {sec:.0f}s", flush=True)
    for f in out:
        out[f][rq.COMBINED] = rq.combine(out[f])
        out[f] = {m: out[f][m] for m in rq.MODELS}
    return out


# --------------------------------------------------------------------------- registry
def code_sha256() -> dict:
    return {f: rq._sha256(os.path.join(config.PKG_DIR, f)) for f in CODE_FILES}


def metadata() -> dict:
    return {"vintage": f"{VINTAGE:%Y-%m-%d}", "git_commit": registry._git_commit(), "uncommitted": rq.uncommitted(),
            "data_sha256": rq._sha256(config.DATA_PATH), "indicators_sha256": rq._sha256(config.INDICATORS_PATH),
            "openai_claims_sha256": rq._sha256(pm.CLAIMS_PATHS["openai"]), "inputs_sha256": rq.inputs_sha256(),
            "code_sha256": code_sha256()}


def frame(results: dict, meta: dict) -> pd.DataFrame:
    parts = []
    for prefix, filings in SETS.items():
        f = rq.to_frame(results[filings])
        f["registry_name"] = prefix + f.model
        f.insert(2, "set", prefix.rstrip("_"))
        f["filings_as_evidence"] = filings
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    out["lower_claim_rule"] = VARIANT
    for k in ("vintage", "git_commit", "data_sha256", "indicators_sha256", "inputs_sha256"):
        out[k] = meta[k]
    out["uncommitted_changes"] = len(meta["uncommitted"])
    return out


def already_registered(path: str = config.REGISTRY_PATH) -> list[str]:
    return [e["model"] for e in registry.load_registry(path)
            if any(str(e.get("model", "")).startswith(p) for p in SETS)]


def register_all(results: dict, meta: dict, path: str = config.REGISTRY_PATH) -> list[dict]:
    """register_summary's entry (written to a scratch file first) plus the sub-task N fields, appended."""
    entries = []
    for prefix, filings in SETS.items():
        for m in rq.MODELS:
            summ, point_only = rq.registry_summary(results[filings][m])
            if not summ:
                print(f"  {prefix}{m}: nothing to register")
                continue
            with tempfile.TemporaryDirectory() as tmp:
                e = registry.register_summary(prefix + m, VINTAGE, summ, path=os.path.join(tmp, "r.jsonl"),
                                              note=NOTES[filings] + (rq.POINT_ONLY_NOTE if point_only else ""))
            e.update({"set": prefix.rstrip("_"), "lower_claim_rule": f"{VARIANT}: " + lower_rule.describe(VARIANT),
                      "filings_as_evidence": filings, "description": rq.MODELS[m],
                      "openai_claims_sha256": meta["openai_claims_sha256"], "inputs_sha256": meta["inputs_sha256"],
                      "code_sha256": meta["code_sha256"], "uncommitted_changes": meta["uncommitted"]})
            entries.append(e)
    with open(path, "a") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return entries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Q4 2026 forecasts v2 (iv115) at vintage 2026-11-13, two sets.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--register", action="store_true")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", default=None, help=f"CSV path (default with --register: {CSV_PATH})")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    if config.USE_FILINGS or config.LOWER_UPPER_RATIO is not None:
        raise SystemExit("defaults expected: config.USE_FILINGS False, config.LOWER_UPPER_RATIO None")
    if a.register and already_registered():
        raise SystemExit(f"already registered: {already_registered()}. The first registration is the record.")
    with open(config.REGISTRY_PATH) as f:
        before = f.read()
    t0 = time.time()
    print(f"Q4 v2 forecasts at {VINTAGE:%Y-%m-%d}, rule {VARIANT}, {a.jobs} job(s):", flush=True)
    results = compute_all(a.jobs)
    meta = metadata()
    out = frame(results, meta)
    for prefix, filings in SETS.items():
        print(f"\n{prefix}: {NOTES[filings]}")
        print(rq.table(out[out.set == prefix.rstrip("_")]))
    print(f"\nRuntime {time.time() - t0:.0f}s.")
    path = a.out or (CSV_PATH if a.register else None)
    if path:
        if a.register and os.path.exists(path):
            raise SystemExit(f"{path} exists; not overwriting")
        out.to_csv(path, index=False)
        print(f"-> {path}")
    if not a.register:
        print("Dry run: nothing registered.")
        return 0
    entries = register_all(results, meta)
    with open(config.REGISTRY_PATH) as f:
        assert f.read().startswith(before), "existing registry lines changed"
    for e in entries:
        print(f"registered {e['model']} at {e['registered_at_utc']}")
    print(f"-> {config.REGISTRY_PATH} ({len(entries)} lines appended)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
