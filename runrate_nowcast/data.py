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
OPTIONAL = {"segment": "total", "indicator": "", "is_forecast": 0, "date_kind": "", "filing": 0}
METRICS = {"run_rate", "revenue", "share", "indicator"}
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
    for c, v in OPTIONAL.items():
        df[c] = df[c].fillna(v) if c in df else v
    df["is_forecast"] = df["is_forecast"].astype(int)
    df["filing"] = df["filing"].astype(int)
    apply_date_windows(df, _fitted_crossing_position())
    _validate(df)
    return df


def load_all(path: str = config.DATA_PATH, indicators: bool = True) -> pd.DataFrame:
    """Claims plus active leading-indicator observations (weekly), in one table."""
    df = load(path)
    if not indicators:
        return df
    from . import altdata
    ind = altdata.indicator_observations()
    if len(ind):
        df = pd.concat([df, ind], ignore_index=True)
    return df


def _fitted_crossing_position():
    """crossing_position from params.json if it has been fitted, else None (config default)."""
    import json
    import os
    if os.path.exists(config.PARAMS_PATH):
        with open(config.PARAMS_PATH) as f:
            return json.load(f).get("crossing_position")
    return None


def apply_date_windows(df: pd.DataFrame, crossing_position: float | None = None) -> None:
    """Where the source wording gives a window [date_lo, date_hi], derive the date from it.

    Most claims ("in August", "October read") are uniform over the window: midpoint,
    sd = width / sqrt(12). Crossing claims (date_kind = crossing: "crossed / passed /
    reached $X") instead put the crossing at a position rho in the window (0 = window
    start, 1 = the announcement), with the sd of a Beta(2 rho, 2 (1 - rho)) position:
    width * sqrt(rho (1 - rho) / 3). At rho = 0.5 this is exactly the uniform rule.
    rho is estimated from the data (see robust.crossing_profile), not assumed.
    """
    if "date_lo" not in df or "date_hi" not in df:
        return
    rho = config.CROSSING_POSITION if crossing_position is None else float(crossing_position)
    lo, hi = pd.to_datetime(df["date_lo"]), pd.to_datetime(df["date_hi"])
    m = lo.notna() & hi.notna()
    width = (hi - lo).dt.days.astype(float)
    cross = m & (df.get("date_kind", pd.Series("", index=df.index)).fillna("") == "crossing")
    uni = m & ~cross
    df.loc[uni, "ref_date"] = lo[uni] + pd.to_timedelta((width[uni] / 2).round(), unit="D")
    df.loc[uni, "date_sd_days"] = width[uni] / 12 ** 0.5
    df.loc[cross, "ref_date"] = lo[cross] + pd.to_timedelta((width[cross] * rho).round(), unit="D")
    df.loc[cross, "date_sd_days"] = width[cross] * (rho * (1 - rho) / 3) ** 0.5


def with_crossing_position(df: pd.DataFrame, rho: float) -> pd.DataFrame:
    """Copy of a loaded table with crossing claims re-dated at position rho."""
    d = df.copy()
    claims = d.metric != "indicator"
    sub = d[claims].copy()
    apply_date_windows(sub, rho)
    d.loc[claims, ["ref_date", "date_sd_days"]] = sub[["ref_date", "date_sd_days"]]
    return d


def _validate(df: pd.DataFrame) -> None:
    if df["obs_id"].duplicated().any():
        raise ValueError(f"duplicate obs_id: {df.loc[df.obs_id.duplicated(), 'obs_id'].tolist()}")
    bad = df[~df.metric.isin(METRICS)]
    if len(bad):
        raise ValueError(f"unknown metric in {bad.obs_id.tolist()}")
    bad = df[~df.kind.isin(KINDS)]
    if len(bad):
        raise ValueError(f"unknown kind in {bad.obs_id.tolist()}")
    bad = df[~df.segment.isin(list(config.SEGMENTS) + ["total"])]
    if len(bad):
        raise ValueError(f"unknown segment in {bad.obs_id.tolist()}")
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
    if row.metric in ("share", "indicator"):
        return row.ref_date          # indicator rows arrive with their lag already applied
    lag = config.DEFINITION_LAG_DAYS[row.definition]
    return row.ref_date - pd.Timedelta(days=round(lag))


def select(df: pd.DataFrame, vintage=None, exclude_ids=(), verbose=False,
           forecasts: bool | None = None, filings: bool | None = None) -> pd.DataFrame:
    """Rows usable as evidence: include==1, published on/before `vintage`, one row per origin.

    Filing rows (S-1 and amendments) are dropped unless filings=True (or config.USE_FILINGS):
    they are evaluation targets. Pass filings=True when selecting targets or scoring.

    Rows sharing an origin_id re-report the same underlying claim. Only the most
    reliable row is kept (lowest tier rank). Its `published` is set to the earliest
    publication in the group, since that is when the information became public.
    """
    d = df[(df.include == 1) & (~df.obs_id.isin(list(exclude_ids)))].copy()
    use_fc = config.USE_FORECASTS if forecasts is None else forecasts
    if not use_fc and "is_forecast" in d:
        d = d[d.is_forecast != 1]
    use_filings = config.USE_FILINGS if filings is None else filings
    if not use_filings and "filing" in d:
        d = d[d.filing.fillna(0).astype(int) == 0]     # 1 = from a filing, 2 = may derive from one
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
