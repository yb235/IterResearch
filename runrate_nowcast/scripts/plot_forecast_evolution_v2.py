# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/plot_forecast_evolution_v2.py
"""Sub-task P item 1: reports/forecast_evolution_v2_2026-11-13.png (+ .csv). See plot_evolution_v2.py."""
import warnings; warnings.simplefilter("ignore")
from runrate_nowcast import plot_evolution_v2

out, png = plot_evolution_v2.run()
print(png)
s = out[out.model == "indicators"].pivot_table(index=["quarter", "vintage"], columns="line", values="mean")
print(s.round(2).to_string())
