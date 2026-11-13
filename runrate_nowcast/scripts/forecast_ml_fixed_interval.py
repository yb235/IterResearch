# Run from the repo root:
#   OMP_NUM_THREADS=1 PYTHONPATH=. python runrate_nowcast/scripts/forecast_ml_fixed_interval.py <variant> [--jobs 4] [--rules publication,period_end]
"""Sub-task N: the corrected ML pipeline (ml_fixed.py, M1 / M2 / SS path and the declared ablations) rerun
with a lower-claim rule (lower_rule.py). Identical to scripts/forecast_ml_fixed.py's run() except for
config.LOWER_UPPER_RATIO, which the worker processes inherit (fork). The claims-only labels
(pm.company_labels), the profiled crossing position and the process-noise refits all run model.Model,
so they all see the rule; anchors (point claims) do not change.

Output: reports/forecast_mlfixed_<variant>_<rule>_2026-11-13.csv (columns as forecast_mlfixed_<rule>_2026-11-10.csv).
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

from runrate_nowcast import config, lower_rule  # noqa: E402
from runrate_nowcast import ml_fixed as mf  # noqa: E402

warnings.simplefilter("ignore")
ASOF = "2026-11-13"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=list(lower_rule.VARIANTS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--rules", default="publication,period_end")
    a = ap.parse_args()
    assert mp.get_start_method() == "fork", "workers must inherit config.LOWER_UPPER_RATIO"
    lower_rule.apply(a.variant)
    mf.clear_caches()
    print(a.variant, lower_rule.describe(a.variant), flush=True)
    for rule in a.rules.split(","):
        out = os.path.join(config.OUTPUT_DIR, f"forecast_mlfixed_{a.variant}_{rule}_{ASOF}.csv")
        if os.path.exists(out):
            raise SystemExit(f"{out} exists; not overwriting")
        t = time.time()
        res = mf.run_backtest(rule, n_jobs=a.jobs)
        res.to_csv(out, index=False)
        print(f"{rule}: {len(res)} rows, {res.target.nunique()} targets, {time.time() - t:.0f}s wall", flush=True)


if __name__ == "__main__":
    main()
