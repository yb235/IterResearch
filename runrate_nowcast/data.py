"""Load, validate and deduplicate the observation table."""
from __future__ import annotations

import warnings

import pandas as pd

from . import config

REQUIRED = [
    "obs_id", "origin_id", "metric", "kind", "value", "ref_date", "period_start",
    "period_end", "definition", "date_sd_days", "published", "tier", "source", "url",
    "include", "notes",
]
METRICS = {"run_rate", "revenue"}
KINDS = {"point", "lower", "upper"}


def load(path: str = config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"obs_id": str, "origin_id": str})
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"observations.csv missing columns: {sorted(missing)}")
    for c in ("ref_date", "period_start", "period_end", "published"):
        df[c] = pd.to_datetime(df[c])
    df["date_sd_days"] = df["date_sd_days"].fillna(0.0).astype(float)
    df["include"] = df["include"].fillna(1).astype(int)
    df["definition"] = df["definition"].fillna("")
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    if df["obs_id"].duplicated().any():
        raise ValueError(f"duplicate obs_id: {df.loc[df.obs_id.duplicated(), 'obs_id'].tolist()}")
    bad = df[~df.metric.isin(METRICS)]
    if len(bad):
        raise ValueError(f"unknown metric in {bad.obs_id.tolist()}")
    bad = df[~df.kind.isin(KINDS)]
    if len(bad):
        raise ValueError(f"unknown kind in {bad.obs_id.tolist()}")
    bad = df[~df.tier.isin(config.TIERS)]
    if len(bad):
        raise ValueError(f"unknown tier in {bad.obs_id.tolist()}")
    rr = df[df.metric == "run_rate"]
    bad = rr[~rr.definition.isin(config.DEFINITION_LAG_DAYS)]
    if len(bad):
        raise ValueError(f"unknown run-rate definition in {bad.obs_id.tolist()}")
    rev = df[df.metric == "revenue"]
    bad = rev[rev.period_start.isna() | rev.period_end.isna()]
    if len(bad):
        raise ValueError(f"revenue rows need period_start/period_end: {bad.obs_id.tolist()}")
    if (df.value <= 0).any():
        raise ValueError("values must be positive ($B)")


def effective_date(row) -> pd.Timestamp:
    """Date of the instantaneous run-rate (or end of the revenue period) a row measures."""
    if row.metric == "revenue":
        return row.period_end
    lag = config.DEFINITION_LAG_DAYS[row.definition]
    return row.ref_date - pd.Timedelta(days=round(lag))


def select(df: pd.DataFrame, vintage=None, exclude_ids=(), verbose=False) -> pd.DataFrame:
    """Rows usable as evidence: include==1, published on/before `vintage`, one row per origin.

    Rows sharing an origin_id re-report the same underlying claim. Only the most
    reliable row is kept (lowest tier rank). Its `published` is set to the earliest
    publication in the group, since that is when the information became public.
    """
    d = df[(df.include == 1) & (~df.obs_id.isin(list(exclude_ids)))].copy()
    if vintage is not None:
        d = d[d.published <= pd.Timestamp(vintage)]
    d["rank"] = d.tier.map(lambda t: config.TIERS[t]["rank"])
    keep = []
    for origin, g in d.groupby("origin_id", sort=False):
        g = g.sort_values(["rank", "published"])
        best = g.iloc[0].copy()
        best["published"] = g.published.min()
        if len(g) > 1 and verbose:
            dropped = g.obs_id.iloc[1:].tolist()
            warnings.warn(f"dedup origin={origin}: keep {best.obs_id}, drop {dropped}")
        keep.append(best)
    if not keep:
        return d.iloc[0:0]
    out = pd.DataFrame(keep).reset_index(drop=True)
    out["t_eff"] = out.apply(effective_date, axis=1)
    extra = out.definition.map(lambda x: config.DEFINITION_EXTRA_DATE_SD.get(x, 0.0))
    out["date_sd_days"] = (out.date_sd_days ** 2 + extra ** 2) ** 0.5
    return out.sort_values(["t_eff", "rank"]).reset_index(drop=True)


def dedup_report(df: pd.DataFrame) -> pd.DataFrame:
    """Rows removed by dedup or by include==0, with the reason."""
    rows = []
    for _, r in df[df.include == 0].iterrows():
        rows.append({"obs_id": r.obs_id, "reason": "include=0", "notes": r.notes})
    d = df[df.include == 1].copy()
    d["rank"] = d.tier.map(lambda t: config.TIERS[t]["rank"])
    for origin, g in d.groupby("origin_id"):
        if len(g) > 1:
            g = g.sort_values(["rank", "published"])
            for oid in g.obs_id.iloc[1:]:
                rows.append({"obs_id": oid, "reason": f"duplicate of {g.obs_id.iloc[0]}",
                             "notes": origin})
    return pd.DataFrame(rows)
