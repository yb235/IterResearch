"""Default settings: source tiers, run-rate definitions, model priors.

Every number here is a starting value. `python -m runrate_nowcast fit` re-estimates
the process noise, and `calibrate` estimates per-tier bias and noise from how each
tier's past claims compare with the audited (GAAP) path. Fitted values are written to
params.json and override these defaults.
"""
from __future__ import annotations

import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PKG_DIR, "data", "observations.csv")
PARAMS_PATH = os.path.join(PKG_DIR, "params.json")
REGISTRY_PATH = os.path.join(PKG_DIR, "forecasts", "registry.jsonl")
OUTPUT_DIR = os.path.join(PKG_DIR, "reports")

MODEL_START = "2025-01-01"
DAYS_PER_YEAR = 365.0
DAYS_PER_MONTH = 365.25 / 12

# Source tiers, from most to least reliable.
#   log_sd: measurement noise. For run-rate claims it is a log sd (0.05 ~ 5%).
#           For revenue claims it is the sd relative to the value.
#   bias:   additive log bias relative to the official definition. The gap between
#           the official definition and GAAP is estimated inside the filter as state b.
TIERS = {
    "gaap":        {"rank": 0, "log_sd": 0.003, "bias": 0.0},
    "official":    {"rank": 1, "log_sd": 0.04,  "bias": 0.0},
    "leak_tier1":  {"rank": 2, "log_sd": 0.08,  "bias": 0.0},
    "leak_single": {"rank": 3, "log_sd": 0.15,  "bias": 0.0},
    "third_party": {"rank": 4, "log_sd": 0.15,  "bias": 0.0},
    "secondary":   {"rank": 5, "log_sd": 0.30,  "bias": 0.0},
}

# How far back each run-rate definition looks. A run-rate quoted on date d is mapped
# to the instantaneous annualized revenue rate at d - lag.
#   t28x13: Anthropic's definition per the S-1 (trailing 28 days of consumption x 13
#           + monthly subscriptions x 12). The 28-day window is centred 14 days back.
DEFINITION_LAG_DAYS = {
    "instant": 0.0,
    "week_x52": 3.5,
    "t28x13": 14.0,
    "month_x12": 15.0,
    "quarter_x4": 45.0,
    "unknown": 7.0,
}
# Extra date uncertainty (days) added for definitions we can't pin down.
DEFINITION_EXTRA_DATE_SD = {"unknown": 10.0}

# Model defaults. Units: l = log(annualized $B); g = log growth per month.
DEFAULT_PARAMS = {
    "sigma_l": 0.004,     # level shock, log per sqrt(day)
    "sigma_g": 0.010,     # growth shock, (log/month) per sqrt(day): ~0.1/month per quarter
    "sigma_b": 0.0005,    # drift in the run-rate-vs-GAAP gap, log per sqrt(day)
    "prior_l": 0.0,       # log(1.0) on 2025-01-01
    "prior_l_sd": 0.5,
    "prior_g": 0.15,
    "prior_g_sd": 0.20,
    "prior_b": 0.0,
    "prior_b_sd": 0.25,
    "growth_reversion": 0.0,   # kappa, per month. 0 = plain local linear trend
    "growth_longrun": 0.03,    # g_bar, log/month (~43%/yr), used only if kappa > 0
    "tiers": TIERS,
}

ALERT_Z = 2.0
