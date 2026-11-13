"""Sub-task P: can the filter learn an indicator's excess growth over revenue? (DATA_PLAN.md, 子任务 P)

Pre-registered (commit 1ff145b) before any run. Variants of the indicator layer:

  baseline  the current model: intercept a_i is a random walk with daily sd `drift`; beta, drift and
            obs_sd are refit by calibrate.fit_indicators at every vintage (catalog priors).
  P1        PRIMARY. The intercept is a local linear trend: a_t = a_{t-1} + b_{t-1}/7 + eta_t,
            b_t = b_{t-1} + zeta_t, with b in log per week, b_0 ~ N(0, 0.01^2) and zeta sd 0.001 per
            week (config.INDICATOR_SLOPE). eta is as now (catalog drift as the prior, refit per vintage).
  P1p       P1': no slope; every indicator's drift fixed at 0.02 (config.INDICATOR_DRIFT, catalog units).
  P1pp      P1'': no slope; beta refit at every vintage (as now) with prior sd 0.5 for every indicator
            (config.INDICATOR_BETA_SD).
  P1p_wk    sensitivity, not pre-registered: P1' with 0.02 per week in calendar units, i.e. a daily sd of
            0.02 / sqrt(7) (see the report on the units of the catalog's `drift`).

Models (the catalog roles switched in this process only, copied from scripts/forecast_interval.py and
scripts/forecast_openrouter.py): B catalog default indicators, E OpenRouter tokens x price (or_price),
G GitHub new-repo share, R Ramp adoption, OC OpenCode npm downloads.
Every run uses the lower-claim rule iv115 (config.LOWER_UPPER_RATIO = 1.15). Evidence sets: "nofil"
(config.USE_FILINGS = False, the package default) and "pub" (filing rows once public, sub-task O).
"""
from __future__ import annotations

import contextlib
import math

import numpy as np
import pandas as pd

from . import altdata, config

ASOF = "2026-11-13"
SLOPE = {"b0_sd": 0.01, "slope_sd": 0.001}
VARIANTS = {
    "baseline": {},
    "P1": {"INDICATOR_SLOPE": SLOPE},
    "P1p": {"INDICATOR_DRIFT": 0.02},
    "P1pp": {"INDICATOR_BETA_SD": 0.5},
    "P1p_wk": {"INDICATOR_DRIFT": 0.02 / math.sqrt(7.0)},
}
PREREGISTERED = ("baseline", "P1", "P1p", "P1pp")
LABELS = {"baseline": "baseline", "P1": "P1 slope", "P1p": "P1' drift 0.02", "P1pp": "P1'' beta sd 0.5",
          "P1p_wk": "P1' drift 0.02/wk (sens.)"}
FLAGS = ("INDICATOR_SLOPE", "INDICATOR_DRIFT", "INDICATOR_BETA_SD")
MODELS = ("B", "E", "G", "R", "OC")
MODEL_LABELS = {"B": "B catalog indicators (SDK)", "E": "E OpenRouter tokens x price", "G": "G GitHub new-repo share",
                "R": "R Ramp adoption", "OC": "OC OpenCode downloads"}
SDK_RAMP = ["npm_anthropic_sdk", "pypi_anthropic", "ramp_token_spend_anthropic"]
INDICATORS = {"B": SDK_RAMP, "E": ["openrouter_anthropic_tokens"], "G": ["github_repo_share_claude"],
              "R": ["ramp_adoption_anthropic"], "OC": ["npm_opencode"]}
EVIDENCE = {"nofil": False, "pub": True}
LOWER = "iv115"


def apply_variant(variant: str) -> None:
    """Set the variant's config flags for the rest of this process (all others back to None)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; one of {sorted(VARIANTS)}")
    for f in FLAGS:
        setattr(config, f, None)
    for k, v in VARIANTS[variant].items():
        setattr(config, k, dict(v) if isinstance(v, dict) else v)


@contextlib.contextmanager
def variant(name: str):
    old = {f: getattr(config, f) for f in FLAGS}
    try:
        apply_variant(name)
        yield
    finally:
        for f, v in old.items():
            setattr(config, f, v)


def patched_catalog(model: str, cat0: pd.DataFrame) -> pd.DataFrame:
    """The catalog with roles as in the model's original script."""
    c = cat0.copy()
    if model == "B":
        return c
    if model == "E":                                   # forecast_openrouter.py or_price
        c.loc[c.indicator == "openrouter_anthropic_tokens", "role"] = "model"
        c.loc[c.indicator.isin(SDK_RAMP), "role"] = "monitor"
    elif model == "OC":                                # forecast_openrouter.py opencode
        c.loc[c.indicator == "npm_opencode", "role"] = "model"
        c.loc[c.indicator.isin(SDK_RAMP), "role"] = "monitor"
    elif model in ("G", "R"):                          # forecast_ramp.py gh_share / ramp_only
        c.loc[c.role == "model", "role"] = "monitor"
        c.loc[c.indicator.isin(INDICATORS[model]), "role"] = "model"
    else:
        raise ValueError(f"unknown model {model!r}")
    return c


_ORIG_CATALOG = altdata.catalog


def use_model(model: str) -> None:
    """Patch altdata.catalog for the rest of this process, as the original scripts do."""
    cat = patched_catalog(model, _ORIG_CATALOG())

    def cat_patched(path=config.CATALOG_PATH):
        return cat.copy()
    altdata.catalog = cat_patched


@contextlib.contextmanager
def model_catalog(model: str):
    old = altdata.catalog
    try:
        use_model(model)
        yield
    finally:
        altdata.catalog = old


def model_frame(model: str) -> pd.DataFrame:
    """Evidence table for a model (call use_model first)."""
    from . import data, pricing
    df = data.load_all()
    if model == "E":
        df = pricing.adjust_indicators(df, 1.0, "github")
    return df


# --------------------------------------------------------------------------- diagnostics
def sign_string(err) -> str:
    """'+' where the forecast was too low (log(actual/forecast) > 0), '-' where too high."""
    return "".join("+" if e > 0 else "-" if e < 0 else "0" for e in err)


def n_runs(err) -> int:
    s = sign_string(err)
    return 0 if not s else 1 + sum(a != b for a, b in zip(s, s[1:]))


def acf(x, k: int = 1) -> float:
    x = np.asarray(x, float)
    n = len(x)
    if n <= k + 1:
        return float("nan")
    d = x - x.mean()
    den = float(d @ d)
    return float(d[k:] @ d[:-k] / den) if den > 0 else float("nan")


def ljung_box(x, lags: int = 3) -> tuple[float, float]:
    """Ljung-Box Q over lags 1..lags and its chi-square p-value."""
    from scipy.stats import chi2
    x = np.asarray(x, float)
    n = len(x)
    if n <= lags + 1:
        return float("nan"), float("nan")
    q = n * (n + 2) * sum(acf(x, k) ** 2 / (n - k) for k in range(1, lags + 1))
    return float(q), float(chi2.sf(q, lags))


def runs_test_p(err) -> float:
    """Wald-Wolfowitz runs test, two-sided normal approximation (few runs = clustered signs)."""
    from scipy.stats import norm
    s = sign_string(err).replace("0", "")
    n1, n2 = s.count("+"), s.count("-")
    n = n1 + n2
    if n1 == 0 or n2 == 0:
        return float("nan")
    r = n_runs([1 if c == "+" else -1 for c in s])
    mu = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n * n * (n - 1))
    return float(2 * norm.sf(abs(r - mu) / math.sqrt(var))) if var > 0 else float("nan")


def posthoc_corrected(bt: pd.DataFrame, published: dict) -> pd.Series:
    """Naive bias correction: subtract from each forecast the mean log error of targets whose actual was
    public at its vintage (published <= vintage). Returns the corrected log errors (0 correction if none)."""
    v = pd.to_datetime(bt.vintage)
    pub = bt.target.map(published)
    out = []
    for i in range(len(bt)):
        known = (pub <= v.iloc[i]) & (bt.target != bt.target.iloc[i])
        c = float(bt.log_err_model[known].mean()) if known.any() else 0.0
        out.append(float(bt.log_err_model.iloc[i]) - c)
    return pd.Series(out, index=bt.index)
