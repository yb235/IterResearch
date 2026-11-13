"""The forecast of a target must not depend on the target's own value (examiner check, 2026-11-13).

Revenue claim noise is proportional to the reported value. observation_predictive used the target's
actual value to size that noise, so tripling the answer changed the predictive sd slightly. It now
scales the noise by the forecast mean.
"""
from __future__ import annotations

import pytest

from runrate_nowcast import backtest, calibrate, data


@pytest.mark.parametrize("tid,rule", [("rev_2026q3", "period_end"), ("rev_2026q2", "publication"),
                                      ("rr_2026_10_s1a", "publication")])
def test_forecast_ignores_target_value(tid, rule):
    df = data.load_all()
    df = df[(df.metric != "indicator") & (df.segment == "total") & (df.metric != "share")]
    p = calibrate.load_params()

    def fc(d):
        full = data.select(d, filings=True)
        row = full[full.obs_id == tid].iloc[0]
        out = backtest.prepublication_forecast(d, row, p, vintage=backtest.target_vintage(row, rule))
        return out[0], out[1]

    changed = df.copy()
    changed.loc[changed.obs_id == tid, "value"] *= 3
    (m0, s0), (m1, s1) = fc(df), fc(changed)
    assert m0 == pytest.approx(m1, abs=1e-12)
    assert s0 == pytest.approx(s1, abs=1e-12)
