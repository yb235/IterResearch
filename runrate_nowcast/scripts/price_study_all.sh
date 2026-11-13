#!/bin/bash
# Run every component backtest of the price study, four at a time, then combine.
# From the repo root: bash runrate_nowcast/scripts/price_study_all.sh [components...]
# A component is VARIANT:GAMMA:RULE; default = all of price_study.COMPONENTS x RULES.
set -e
if [ $# -eq 0 ]; then
  set -- $(python -c "from runrate_nowcast import price_study as p; print(' '.join(f'{v}:{g:g}:{r}' for v, g in p.COMPONENTS for r in p.RULES))")
fi
printf '%s\n' "$@" | xargs -P 4 -I{} bash -c 'IFS=: read v g r <<< "{}"; python -m runrate_nowcast price-study run --variant $v --gamma $g --rule $r'
python -m runrate_nowcast price-study combine > /dev/null
echo done
