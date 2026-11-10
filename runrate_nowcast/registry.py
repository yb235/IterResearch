"""Pre-registration: time-stamp forecasts before a disclosure, then score them afterward.

register() appends one JSON line to forecasts/registry.jsonl. The line holds:
  - revenue forecasts for every period that is not yet disclosed (quarter, year, month)
  - a weekly fan for the run-rate on both the official and GAAP bases over the next 120
    days, so any later run-rate claim can be scored whatever date it lands on
  - the git commit and a sha256 of the data file, which pin what the model knew
Commit registry.jsonl right away; the git history is the time-stamp of record.
score() matches registered forecasts with claims published after registration.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config, data, report
from .model import period_key


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.PKG_DIR,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _sha256(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def register(df, params, asof, note: str = "", path: str = config.REGISTRY_PATH,
             data_path: str = config.DATA_PATH, horizon_days: int = 120) -> dict:
    asof = pd.Timestamp(asof)
    model, res, sel = report.run_nowcast(df, params, asof)
    fan_end = asof + pd.Timedelta(days=horizon_days)
    res2 = model.run(sel, end=fan_end, queries=report.default_queries(asof))
    disclosed = {period_key(r.period_start, r.period_end)
                 for r in sel[(sel.metric == "revenue") & (sel.tier == "gaap")].itertuples()}
    rev = {}
    for p in res2.periods:
        if p.key in disclosed:
            continue
        r = res2.period_revenue(p.key)
        rev[p.key] = {k: round(v, 4) for k, v in r.items() if k != "complete"}
    fan = []
    for d in pd.date_range(asof - pd.Timedelta(days=28), fan_end, freq="7D"):
        mo, so = res2.log_runrate(d, "t28x13")
        mi, si = res2.log_runrate(d, "instant")
        fan.append({"date": f"{d:%Y-%m-%d}", "official_log_mu": round(mo, 5), "official_log_sd": round(so, 5),
                    "instant_log_mu": round(mi, 5), "instant_log_sd": round(si, 5)})
    entry = {
        "registered_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "asof": f"{asof:%Y-%m-%d}",
        "note": note,
        "git_commit": _git_commit(),
        "data_sha256": _sha256(data_path),
        "n_obs": int(len(sel)),
        "params": {k: model.p[k] for k in ("sigma_l", "sigma_g", "sigma_b", "growth_reversion", "growth_longrun")},
        "revenue": rev,
        "runrate_fan": fan,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load_registry(path: str = config.REGISTRY_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score(df, params=None, path: str = config.REGISTRY_PATH) -> pd.DataFrame:
    """Score registered forecasts against claims published after each registration."""
    from .model import Model
    tiers = Model(params).p["tiers"]
    sel = data.select(df)
    rows = []
    for e in load_registry(path):
        reg_day = pd.Timestamp(e["registered_at_utc"][:10])
        later = sel[sel.published > reg_day]
        fan = pd.DataFrame(e["runrate_fan"])
        fan["date"] = pd.to_datetime(fan.date)
        x = (fan.date - fan.date.min()).dt.days.to_numpy(float)
        for r in later.itertuples():
            if r.metric == "revenue":
                key = period_key(r.period_start, r.period_end)
                if key not in e["revenue"]:
                    continue
                f = e["revenue"][key]
                sd = math.sqrt(f["sd"] ** 2 + (tiers[r.tier]["log_sd"] * r.value) ** 2)
                y, mu = r.value, f["mean"]
            else:
                xr = (pd.Timestamp(r.ref_date) - fan.date.min()).days
                if not (x[0] <= xr <= x[-1]):
                    continue
                basis = "instant" if r.definition == "instant" else "official"
                mu = float(np.interp(xr, x, fan[f"{basis}_log_mu"]))
                s0 = float(np.interp(xr, x, fan[f"{basis}_log_sd"]))
                sd = math.sqrt(s0 ** 2 + tiers[r.tier]["log_sd"] ** 2)
                mu += tiers[r.tier].get("bias", 0.0)
                y = math.log(r.value)
            z = (y - mu) / sd
            if r.kind == "point":
                pit = float(norm.cdf(z))
                rows.append({"registered": e["registered_at_utc"], "obs_id": r.obs_id, "kind": r.kind,
                             "z": z, "pit": pit, "in80": abs(z) < 1.2816})
            else:
                p_true = float(norm.sf(z)) if r.kind == "lower" else float(norm.cdf(z))
                rows.append({"registered": e["registered_at_utc"], "obs_id": r.obs_id, "kind": r.kind,
                             "z": z, "p_claim_true": p_true})
    return pd.DataFrame(rows)
