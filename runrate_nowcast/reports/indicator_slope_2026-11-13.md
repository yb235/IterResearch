# Sub-task P: can the filter learn the indicators' excess growth? And the evolution chart (2026-11-13)

> **审核更正（2026-11-13，依据独立审核 `reviews/fable_review_2026-11-13_P_5702655.md`，协调者已重算第 1、2（合并后的 Ljung-Box）、3、4、6、7 条；第 5 条的分组 bootstrap 是审核者的计算，协调者未重算）。下文正文保留执行者原文；与以下更正冲突处，以更正为准。**
>
> 1. **§6 关于 NYT “over $100B” 的那句是错的。** `rr_2026_09_nyt100` 的 `include = 0`（因误引被排除），`fc_nyt_2026_09` 是前瞻说法，`USE_FORECASTS = False` 时也不进证据。2026-08-17 之后到 11-06（WSJ“超过 80”）之前，不用 S-1 的证据里没有任何新说法。A 在 10/31 预测 112（真实 81）完全是增长外推。
> 2. **一步预测不能按 15 个、8 个独立观测来数。** 不用 S-1 时，10/31–11/03 的 3 个数字共享“停在 8/17”的信息，11/10 的 2 个数字共享“另加 WSJ”的信息，10/26 的 6 个 S-1 数共享同一状态。把同一时点合并后，B 的 Ljung-Box p 为 0.30（原模型）和 0.35（P1），不显著。开头结论 2 引用的 0.017 → 0.046 不应再用。用已公开 S-1 的一组（能真正检验“更新后是否还偏”）：B 每次更新后仍偏高，但幅度在缩小（z 从 −3.3 到 −1.1/−1.6），Ljung-Box p 0.67/0.93。结论“P1 没有修掉 B 的高估”成立，但证据强度远低于原文写法。
> 3. **“P1 在 G、R 上引入了序列相关”应改为“期末时点上 G、R 退回到只用说法的 A 的误差模式”。** 与 A 的误差相关：G 0.19 → 0.81，R 0.17 → 0.92（协调者已重算）。其他时点和一步预测里没有这种相关。
> 4. **OC“误差减半”的 86% 来自 `rev_2025fy` 一个目标**（原模型预测 2777，P1 6.97，真实 4.59；去掉它是 0.767 对 0.688，协调者已重算）。“后期斜率吸收”的机制说法不成立，删除。
> 5. **B 的配对改善（0.012）按信息状态分组 bootstrap 后区间跨 0**，且受优化起点影响。应读作“很小，与 0 无法区分”。
> 6. **参数通道不能说“各版本都受影响所以公平”。** 把起点换成 `unfitted_params()` 后，原模型对 10/31 run-rate 的预测从 120 变成 107，P1 从 111.3 变成 110.9。开头“120 对 111”的例子方向随之反转。
> 7. **§8 第 3 条不实。** 0.79 是 Pearson 估计量（本报告 summary CSV 的 `acf1_pearson` = 0.789），0.59 是标准估计量，两者都对。
> 8. **其他：**
>    - “大多数时点还没有更早的真实值公开”与 CSV 矛盾：`n_corrected` 为 11/13 和 14/15。
>    - P1″ 并非处处在 ±0.016 内：R 期末时点 +0.034。
>    - P1′ 的标签：按字面登记的是 0.02/周（本文的 “P1′ wk”），0.02/√天是执行者的解读。
>    - 登记写“a 的噪声沿用目录 drift”，实际仍按时点重估。秋季 SDK drift 被估到目录值约 4 倍，会和斜率争着吸收超额增长。
> 9. **新图已按审核意见修正**（`forecast_evolution_v2_2026-11-13.png`）：
>    - 图例改为“按规则可用的已公开数据”；
>    - 脚注列出已公开但不作输入的行；
>    - 写明 8/17–11/01 没有新说法进入证据；
>    - Ramp 数据 11/10 才公开、图中未用；
>    - 图用全样本参数，严格回测里 11/01 的 Q3 预测是 23.0，不是图上的 20.1。
>
> **P1 不是可用的修复，三个开关保持默认 None。**


Pre-registered in DATA_PLAN.md (子任务 P, commit 1ff145b) before any run. Errors are log(actual / forecast):
**negative = forecast too high**. All runs: lower claims as [X, 1.15X] (iv115), parameters refit at every vintage
(process noise, then beta / drift / obs_sd of every indicator), crossing position profiled, `check_no_leak` on plus a
raw-table recorder. Both vintage rules, both evidence sets (nofil = S-1 numbers never inputs; pub = filing rows
once public). 90 backtests (5 models x 4 pre-registered variants x 2 rules x 2 evidence sets, plus one
sensitivity variant without filings), every one completed; nothing was skipped.

## Answer first

1. **Chart fixed** (`forecast_evolution_v2_2026-11-13.png`). For each model, solid = every row public on the
   date, dashed = the same rows minus direct reports of the plotted quarter's revenue. Q2 2026 on 2026-08-17
   (Bloomberg's leak of $11.5B): + indicators 11.63 with the leak vs 13.09 without; claims only 11.61 vs 13.02
   (actual 11.555). Q3 2026: no direct report was public before the S-1/A, so the two lines coincide (stated in the
   footnote).
2. **P1 does not remove the persistent overestimate of the SDK model (B), which is what the user complained
   about.** Period_end vintage, no filings: 11 of 13 targets too high under baseline and under P1. The mean signed
   error moves from -0.161 to -0.146. The sign string `----++-------` and the lag-1 autocorrelation (+0.59) are
   identical. The paired gain in |log error| is real but tiny: 0.012 (90% CI 0.002 to 0.021).
   One-step-ahead: every point claim that arrived from late October 2026 on was over-predicted by 2.0 to 3.6 sd
   under every variant. Example: the 10/31 run-rate forecast the day before publication was 120 (baseline),
   111 (P1), 109 (P1') and 119 (P1''); the actual was 81. Mean z of the claims arriving from May 2026: -1.05
   baseline, -0.81 P1. The Ljung-Box p-value (lags 1-3) moves from 0.017 to 0.046, so the serial correlation is
   weaker but still there.
3. **Where P1 moves things, it is mostly by other mechanisms, and it adds serial correlation in two models.**
   - OC: |log error| halves (period_end 1.20 -> 0.67). The learned slope (about +0.02 log/week by Oct 2026)
     absorbs OpenCode's excess growth, so at late vintages the process-noise refit no longer makes revenue
     growth explode. OC is still unusable: 10 of 13 too high at period_end, mean -0.57.
   - R: removes R's under-prediction at period_end (+0.40 -> -0.04, |err| 0.42 -> 0.13). But R becomes 9 of 13
     too high with a clustered sign string (`----++++-----`, acf1 +0.49), and 80% coverage under the publication
     rule falls from 0.93 to 0.67.
   - G: |err| falls (0.104 -> 0.071, CI crosses 0), but G flips from under- to over-prediction (3 -> 10 of 13 too
     high) and becomes serially correlated (acf1 -0.07 -> +0.57).
   - E: no change.
4. **P1' (drift 0.02) is a little better than P1 for B** (period_end |err| 0.181, 10 of 13 too high), with the
   same sign pattern. It costs E its calibration (80% coverage 0.92 -> 0.69). **P1'' (beta prior sd 0.5) changes
   nothing that matters** (B: -0.001).
5. **Naive post-hoc correction** (subtract the mean error of targets already published at the vintage): B
   period_end |err| 0.202 -> 0.191, publication 0.139 -> 0.160 (worse). This is as expected: most actuals were
   published together on 10/26-11/10, after most vintages.
6. **Part of the stated diagnosis was wrong.** See section 6: the betas and drifts are re-estimated at every
   vintage, and the SDK drift is refit to about 4x its catalog value. The SDK slope that P1 can learn is small and
   uncertain because the excess growth started only in H2 2026. The OpenCode failure is mainly the process-noise
   refit, not the beta.

## 1. What was run, and how the spec was read

| Variant | What changes (opt-in config flag; default None = previous code, bit-identical) |
|---|---|
| baseline | current model (sub-task N's iv115 runs; reproduced bit-for-bit for B, E, G, R) |
| P1 (primary) | `config.INDICATOR_SLOPE = dict(b0_sd=0.01, slope_sd=0.001)`: each indicator intercept is a local linear trend. Daily steps: a_t = a_{t-1} + b_{t-1}/7 + eta_t (eta daily sd = the indicator's drift, refit per vintage as now), b_t = b_{t-1} + zeta_t; b in log per week, b_0 ~ N(0, 0.01^2), var(zeta) per day = 0.001^2/7 (0.001 per week) |
| P1' | `config.INDICATOR_DRIFT = 0.02`: every indicator's drift fixed at 0.02 and not refit; beta and obs_sd still refit |
| P1'' | `config.INDICATOR_BETA_SD = 0.5`: beta prior sd 0.5 for every indicator in the per-vintage refit (catalog: SDK 0.25, Ramp tokens 0.3, others 1.0) |
| P1' wk (sensitivity, not pre-registered) | drift fixed at 0.02/sqrt(7) per day = 0.02 per calendar week |

Readings I had to make (flag for the examiner):
- **Units.** The filter steps daily, and the catalog's `drift` is a log sd per sqrt(day). The pre-registration
  calls the catalog values "0.3%/week (SDK), 1%/week (OpenCode)"; in calendar terms they are 0.79% and 2.6% per
  week. For P1 I used calendar weeks as written: the slope is an excess growth rate per week, and only that
  reading is plausible (0.01 per day would mean 38x a year). For P1' I used the catalog's own units: 0.02 per
  sqrt(day), i.e. 5.3% per week. That is the only reading that "raises" the drift for every indicator, as the
  spec intends; 0.02 per calendar week would lower it for OpenCode, OpenRouter and GitHub (0.026 per week now).
  The literal calendar reading is reported as the sensitivity "P1' wk". It does not help either: B's |err| falls
  by 0.015, but 12 of 13 targets are too high (mean -0.173).
- **P1' fixes the drift.** With refit on, a drift that is only a prior centre would be refit away (log-sd 1
  prior), so P1' takes 0.02 as fixed.
- **P1''.** `calibrate.fit_indicators` already refits beta at every vintage for every indicator with role
  `model` that has rows in the evidence (`names = [k for k, v in specs.items() if v.get("role") == "model" and
  k in set(sel.indicator)]`). The only betas never refit belong to indicators with no row public at the vintage:
  Ramp token spend in B (first public 2026-11-10, after every vintage), and OpenRouter before November 2025. So
  P1'' changes only the prior sd.
- **Models:** B = catalog defaults (npm SDK, PyPI SDK, Ramp tokens; in practice npm and PyPI only), E = OpenRouter
  tokens x price, G = GitHub new-repo share, R = Ramp adoption, OC = OpenCode npm downloads. Catalog patches and
  frames are copied from `forecast_interval.py` / `forecast_openrouter.py`.

## 2. The evolution chart

`forecast_evolution_v2_2026-11-13.png` (+ `.csv`) is drawn by the new `plot_evolution_v2.py`;
`plot_backtest.py` is untouched. It uses the current model with lower claims as [X, 1.15X]. The models are
claims-only (A frame) and + indicators (B frame), as in the old chart. Vintages: the old weekly grid plus every day
a run-rate or revenue claim was published, so each jump sits on its news day. Parameters: params.json (full
sample), as in the old chart. The footnote now says this is the only input that is not as of the date, and that
S-1 numbers are never inputs.

## 3. Main tables

Each cell: mean signed log error / share too high / mean |log error| / 80% coverage. Then the signs in vintage
order (`-` = too high; ties broken by the target's date), the number of same-sign runs, and the lag-1
autocorrelation (standard estimator; Pearson version in the CSV). Common targets: 15 (publication) and 13
(period_end), the same for every variant (checked).

### No filings (nofil)

| model | variant | publication (n=15) | publication signs, runs, acf1 | period_end (n=13) | period_end signs, runs, acf1 |
|:--|:--|:--|:--|:--|:--|
| B | baseline | -0.067 / 53% / 0.139 / 0.60 | `-+-+--+++++----` 7, +0.35 | -0.161 / 85% / 0.202 / 0.62 | `----++-------` 3, +0.59 |
| B | P1 slope | -0.055 / 53% / 0.129 / 0.60 | `-+-+--+++++----` 7, +0.26 | -0.146 / 85% / 0.190 / 0.62 | `----++-------` 3, +0.59 |
| B | P1' drift 0.02 | -0.045 / 47% / 0.126 / 0.60 | `-+-++-+++++----` 7, +0.26 | -0.125 / 77% / 0.181 / 0.62 | `----+++------` 3, +0.63 |
| B | P1'' beta sd 0.5 | -0.066 / 53% / 0.138 / 0.60 | `-+-+--+++++----` 7, +0.34 | -0.160 / 85% / 0.201 / 0.62 | `----++-------` 3, +0.59 |
| B | P1' wk (sens.) | -0.076 / 60% / 0.137 / 0.67 | `---+--+++++----` 5, +0.47 | -0.173 / 92% / 0.187 / 0.69 | `-----+-------` 3, +0.55 |
| E | baseline | +0.046 / 27% / 0.097 / 0.87 | `-+-++++++++-++-` 7, -0.35 | -0.000 / 54% / 0.121 / 0.92 | `----+++-+-++-` 7, -0.20 |
| E | P1 slope | +0.036 / 27% / 0.082 / 0.93 | `-+-++++++++-++-` 7, -0.38 | -0.006 / 54% / 0.120 / 0.92 | `----+++-+-++-` 7, -0.10 |
| E | P1' drift 0.02 | +0.048 / 33% / 0.100 / 0.73 | `-+-++-+++++-++-` 9, -0.40 | +0.011 / 54% / 0.123 / 0.69 | `----+++-+-++-` 7, -0.08 |
| E | P1'' beta sd 0.5 | +0.043 / 27% / 0.093 / 0.87 | `-+-++++++++-++-` 7, -0.30 | -0.005 / 54% / 0.119 / 0.92 | `----+++-+-++-` 7, -0.22 |
| G | baseline | +0.019 / 40% / 0.069 / 0.93 | `++-+++-++--++--` 8, +0.24 | +0.086 / 23% / 0.104 / 0.92 | `-+-++++++++-+` 6, -0.07 |
| G | P1 slope | -0.016 / 60% / 0.067 / 0.87 | `-+-++--++--+---` 9, +0.09 | -0.018 / 77% / 0.071 / 0.92 | `----+++------` 3, +0.57 |
| G | P1' drift 0.02 | +0.003 / 60% / 0.076 / 0.80 | `-+-++--++--+---` 9, -0.17 | +0.016 / 62% / 0.084 / 0.85 | `--+-+++--+---` 7, +0.40 |
| G | P1'' beta sd 0.5 | +0.021 / 40% / 0.072 / 0.93 | `++-+++-++--++--` 8, +0.23 | +0.104 / 23% / 0.119 / 0.85 | `-+-++++++++-+` 6, -0.10 |
| R | baseline | +0.073 / 27% / 0.113 / 0.93 | `++-+++-++++++--` 6, +0.01 | +0.403 / 15% / 0.421 / 0.62 | `++++++++-++-+` 5, -0.35 |
| R | P1 slope | +0.011 / 47% / 0.117 / 0.67 | `-+-++-+++++----` 7, -0.23 | -0.036 / 69% / 0.127 / 0.77 | `----++++-----` 3, +0.49 |
| R | P1' drift 0.02 | +0.048 / 33% / 0.118 / 0.80 | `++-+++-+++++---` 6, +0.19 | +0.134 / 23% / 0.231 / 0.85 | `++++++++++---` 2, +0.46 |
| R | P1'' beta sd 0.5 | +0.075 / 27% / 0.114 / 0.93 | `++-+++-++++++--` 6, +0.01 | +0.441 / 15% / 0.455 / 0.62 | `++++++++-++-+` 5, -0.36 |
| OC | baseline | +0.161 / 47% / 0.403 / 0.60 | `--+++-+++++----` 5, +0.56 | -1.132 / 77% / 1.200 / 0.77 | `-----++---+--` 5, -0.17 |
| OC | P1 slope | +0.038 / 40% / 0.297 / 0.73 | `--+++-++++--++-` 7, +0.35 | -0.567 / 77% / 0.667 / 0.77 | `-----++---+--` 5, -0.37 |
| OC | P1' drift 0.02 | +0.114 / 40% / 0.351 / 0.67 | `--+++-+++++-+--` 7, +0.51 | -0.653 / 77% / 0.722 / 0.77 | `-----++---+--` 5, -0.39 |
| OC | P1'' beta sd 0.5 | +0.163 / 40% / 0.418 / 0.60 | `--+++-+++++--+-` 7, +0.56 | -1.151 / 77% / 1.216 / 0.77 | `-----++---+--` 5, -0.17 |

For reference: claims-only A (sub-task N, iv115) is at -0.005 / 47% / 0.124 / 0.73 (publication) and
-0.055 / 69% / 0.165 / 0.85 (period_end). Its 10/31 forecast at the period_end vintage is 112 (actual 81).

### Filings allowed once public (pub)

| model | variant | publication (n=15) | publication signs, runs, acf1 | period_end (n=13) | period_end signs, runs, acf1 |
|:--|:--|:--|:--|:--|:--|
| B | baseline | -0.032 / 60% / 0.089 / 0.67 | `-+-+---++++----` 7, -0.08 | -0.137 / 85% / 0.179 / 0.62 | `----++-------` 3, +0.64 |
| B | P1 slope | -0.023 / 60% / 0.082 / 0.73 | `-+-+---++++----` 7, -0.17 | -0.119 / 85% / 0.163 / 0.62 | `----++-------` 3, +0.62 |
| B | P1' drift 0.02 | -0.011 / 47% / 0.079 / 0.73 | `-+-++-+++++----` 7, -0.28 | -0.097 / 77% / 0.153 / 0.62 | `----+++------` 3, +0.67 |
| B | P1'' beta sd 0.5 | -0.032 / 60% / 0.088 / 0.67 | `-+-+---++++----` 7, -0.09 | -0.136 / 85% / 0.178 / 0.62 | `----++-------` 3, +0.64 |
| E | baseline | +0.034 / 27% / 0.065 / 1.00 | `-+-++++++++-++-` 7, -0.17 | +0.018 / 54% / 0.100 / 0.92 | `----+++-+-++-` 7, -0.07 |
| E | P1 slope | +0.035 / 27% / 0.069 / 0.93 | `-+-++++++++-++-` 7, -0.45 | +0.014 / 54% / 0.097 / 0.92 | `----+++-+-++-` 7, +0.02 |
| E | P1' drift 0.02 | +0.031 / 33% / 0.082 / 0.73 | `-+-++-+++++-++-` 9, -0.39 | +0.022 / 54% / 0.110 / 0.77 | `----+++-+-++-` 7, +0.05 |
| E | P1'' beta sd 0.5 | +0.032 / 27% / 0.062 / 1.00 | `-+-++++++++-++-` 7, -0.03 | +0.014 / 54% / 0.097 / 0.92 | `----+++-+-++-` 7, -0.09 |
| G | baseline | +0.042 / 33% / 0.068 / 0.93 | `++-+++-+++-+--+` 9, +0.08 | +0.094 / 23% / 0.113 / 0.92 | `-+-++++++++-+` 6, -0.06 |
| G | P1 slope | +0.019 / 40% / 0.055 / 0.87 | `-+-++-++++-+--+` 10, -0.18 | -0.012 / 69% / 0.075 / 0.92 | `----+++-----+` 4, +0.35 |
| G | P1' drift 0.02 | +0.032 / 40% / 0.069 / 0.80 | `-+-++-++++-+--+` 10, -0.46 | +0.027 / 54% / 0.087 / 0.85 | `--+-+++--+--+` 8, +0.21 |
| G | P1'' beta sd 0.5 | +0.044 / 33% / 0.071 / 0.93 | `++-+++-+++-+--+` 9, +0.09 | +0.112 / 23% / 0.127 / 0.85 | `-+-++++++++-+` 6, -0.07 |
| R | baseline | +0.070 / 33% / 0.106 / 0.93 | `++-+++-++++---+` 7, +0.02 | +0.405 / 15% / 0.422 / 0.62 | `++++++++-++-+` 5, -0.35 |
| R | P1 slope | +0.026 / 40% / 0.098 / 0.73 | `-+-++-+++++---+` 8, -0.48 | -0.022 / 62% / 0.121 / 0.77 | `----++++----+` 4, +0.40 |
| R | P1' drift 0.02 | +0.072 / 27% / 0.098 / 0.87 | `++-+++-+++++--+` 7, -0.01 | +0.154 / 15% / 0.219 / 0.85 | `++++++++++--+` 3, +0.36 |
| R | P1'' beta sd 0.5 | +0.071 / 33% / 0.107 / 0.93 | `++-+++-++++---+` 7, +0.02 | +0.443 / 15% / 0.456 / 0.62 | `++++++++-++-+` 5, -0.36 |
| OC | baseline | +0.094 / 40% / 0.397 / 0.67 | `--+++-+++++--+-` 7, +0.45 | -1.098 / 77% / 1.193 / 0.77 | `-----++-----+` 4, -0.16 |
| OC | P1 slope | -0.058 / 47% / 0.189 / 0.93 | `--+++--+-++-++-` 9, +0.27 | -0.575 / 69% / 0.661 / 0.85 | `-----++---+-+` 6, -0.36 |
| OC | P1' drift 0.02 | -0.027 / 47% / 0.235 / 0.80 | `--+++--++++-+--` 7, +0.34 | -0.616 / 69% / 0.698 / 0.77 | `-----++---+-+` 6, -0.35 |
| OC | P1'' beta sd 0.5 | +0.087 / 40% / 0.403 / 0.67 | `--+++-+++++--+-` 7, +0.45 | -1.116 / 77% / 1.208 / 0.77 | `-----++-----+` 4, -0.16 |

Targets without any filing row in their pub evidence reproduce the nofil forecasts exactly, for every model,
variant and rule (tested).

**Naive post-hoc correction** (`indicator_slope.posthoc_corrected`; columns `posthoc_*` of the summary CSV). It
barely moves anything: B nofil |err| 0.202 -> 0.191 (period_end) and 0.139 -> 0.160 (publication, worse); B
pub 0.179 -> 0.167 and 0.089 -> 0.105. At most vintages no earlier target's actual was public yet.

## 4. Paired comparisons, P1 vs baseline

Mean change in |log error| (negative = P1 better) and in signed error (positive = P1 lower), 10000 bootstrap
draws over targets, 90% intervals. P1', P1'' and the sensitivity variant are in `indicator_slope_paired_2026-11-13.csv`.

| model | evidence | rule | n | abs. error change (90% CI) | better/worse | signed change (90% CI) |
|:--|:--|:--|--:|:--|:--|:--|
| B | nofil | period_end | 13 | -0.012 (-0.021, -0.002) | 9/4 | +0.015 (+0.006, +0.024) |
| B | nofil | publication | 15 | -0.010 (-0.020, -0.000) | 10/5 | +0.012 (+0.003, +0.022) |
| E | nofil | period_end | 13 | -0.002 (-0.014, +0.011) | 5/6 | -0.006 (-0.018, +0.007) |
| E | nofil | publication | 15 | -0.015 (-0.042, +0.006) | 10/4 | -0.010 (-0.037, +0.012) |
| G | nofil | period_end | 13 | -0.034 (-0.083, +0.005) | 8/5 | -0.105 (-0.163, -0.057) |
| G | nofil | publication | 15 | -0.002 (-0.017, +0.013) | 7/8 | -0.036 (-0.059, -0.014) |
| R | nofil | period_end | 13 | -0.294 (-0.526, -0.082) | 8/5 | -0.439 (-0.667, -0.236) |
| R | nofil | publication | 15 | +0.004 (-0.030, +0.037) | 6/9 | -0.062 (-0.107, -0.020) |
| OC | nofil | period_end | 13 | -0.533 (-1.421, -0.028) | 10/3 | +0.566 (+0.071, +1.451) |
| OC | nofil | publication | 15 | -0.106 (-0.227, -0.001) | 10/5 | -0.123 (-0.243, -0.017) |
| B | pub | period_end | 13 | -0.015 (-0.029, -0.003) | 9/4 | +0.018 (+0.007, +0.031) |
| B | pub | publication | 15 | -0.007 (-0.014, +0.001) | 11/4 | +0.009 (+0.002, +0.016) |
| E | pub | period_end | 13 | -0.003 (-0.015, +0.009) | 5/6 | -0.004 (-0.016, +0.008) |
| E | pub | publication | 15 | +0.004 (-0.004, +0.013) | 8/6 | +0.001 (-0.008, +0.010) |
| G | pub | period_end | 13 | -0.038 (-0.087, +0.002) | 10/3 | -0.106 (-0.165, -0.058) |
| G | pub | publication | 15 | -0.013 (-0.027, -0.001) | 10/5 | -0.023 (-0.048, -0.001) |
| R | pub | period_end | 13 | -0.301 (-0.532, -0.090) | 9/4 | -0.427 (-0.657, -0.221) |
| R | pub | publication | 15 | -0.008 (-0.042, +0.025) | 7/8 | -0.044 (-0.092, +0.001) |
| OC | pub | period_end | 13 | -0.532 (-1.420, -0.037) | 11/2 | +0.523 (+0.028, +1.414) |
| OC | pub | publication | 15 | -0.208 (-0.373, -0.068) | 11/4 | -0.152 (-0.337, +0.015) |

Other variants against the baseline, in brief: P1' on B -0.021 (-0.041, -0.003) at period_end without filings.
P1'' is within +/-0.016 everywhere: slightly worse for G, R and OC (intervals above zero), B -0.001. The P1' wk
sensitivity is worse for E, G and R.

## 5. One-step-ahead: every point claim as it arrives (does the bias persist?)

There are 22 point claims of total run-rate or revenue (non-forecast, all tiers; filings are claims to predict).
The 20 with enough evidence are taken in order of publication. Each is forecast from the rows public the day
before (own origin excluded, `check_no_leak` on, crossing position profiled at that date), with the parameters of
the latest backtest refit on or before that date (`params_*.json`; catalog priors before the first refit).
z = (actual - forecast) / sd. Six S-1 numbers arrive on the same day (10/26) and are predicted from the same
information, so the autocorrelation and Ljung-Box statistics treat a tied cluster as serial. Read them as
descriptive. Full table: `indicator_slope_onestep_2026-11-13.csv`; per claim: `..._onestep_long_...csv`; chart:
`indicator_slope_onestep_2026-11-13.png`.

| model | variant | evidence | mean z | share z<0 | acf1 | Ljung-Box p (1-3) | mean z before May 2026 | mean z from May 2026 (15 claims) | mean log err from May 2026 | signs |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| B | baseline | nofil | -0.79 | 0.60 | 0.48 | 0.02 | -0.01 | -1.05 | -0.18 | `-+-+----++++++------` |
| B | P1 | nofil | -0.61 | 0.60 | 0.40 | 0.05 | 0.00 | -0.81 | -0.16 | `-+-+----++++++------` |
| B | P1' | nofil | -0.53 | 0.55 | 0.37 | 0.05 | 0.16 | -0.76 | -0.15 | `-+-++---++++++------` |
| B | P1'' | nofil | -0.78 | 0.60 | 0.48 | 0.02 | -0.01 | -1.04 | -0.18 | `-+-+----++++++------` |
| E | baseline | nofil | 0.22 | 0.35 | -0.42 | 0.05 | 0.17 | 0.24 | -0.00 | `-+-++++-++++++--+-+-` |
| E | P1 | nofil | 0.13 | 0.40 | -0.25 | 0.14 | 0.23 | 0.10 | -0.02 | `-+-+++--++++++--+-+-` |
| G | baseline | nofil | 0.07 | 0.45 | -0.02 | 0.65 | 0.49 | -0.07 | 0.01 | `++-+++++--++--+-+---` |
| G | P1 | nofil | -0.18 | 0.65 | -0.06 | 0.85 | 0.21 | -0.31 | -0.06 | `-+-++-+---++--+-----` |
| R | baseline | nofil | 0.69 | 0.35 | -0.07 | 0.85 | 0.72 | 0.68 | 0.11 | `++-+++++--+++++-+---` |
| R | P1 | nofil | 0.05 | 0.50 | -0.02 | 0.34 | 0.49 | -0.10 | -0.04 | `-+-++-+-++++++------` |
| OC | baseline | nofil | 1.21 | 0.55 | 0.64 | 0.01 | -0.63 | 1.82 | -0.02 | `--+++---++++++------` |
| OC | P1 | nofil | 0.19 | 0.50 | 0.30 | 0.05 | -0.22 | 0.33 | -0.08 | `--+++-+--++++---+-+-` |
| B | baseline | pub | -0.53 | 0.65 | 0.22 | 0.67 | -0.01 | -0.70 | -0.11 | `-+-+----+-++++------` |
| B | P1 | pub | -0.38 | 0.65 | 0.08 | 0.93 | 0.00 | -0.51 | -0.09 | `-+-+----+-++++------` |
| OC | baseline | pub | 0.26 | 0.45 | 0.43 | 0.08 | -0.63 | 0.56 | -0.06 | `--+++---++++++---++-` |
| OC | P1 | pub | -0.09 | 0.55 | 0.12 | 0.28 | -0.22 | -0.05 | -0.18 | `--+++-+---+-++--+-+-` |

B, nofil, the claims that arrived from 10/31 on (forecast made the day before; z in brackets):

| claim (published) | actual | baseline | P1 | P1' | P1'' |
|:--|--:|--:|--:|--:|--:|
| Sacra ~70 (10/31) | 70 | 148 (-3.5) | 142 (-3.0) | 140 (-2.8) | 147 (-3.5) |
| Q3 revenue, S-1/A (11/02) | 16.96 | 23.0 (-2.4) | 22.5 (-2.0) | 22.6 (-2.0) | 22.9 (-2.4) |
| SemiAnalysis ~74 (11/03) | 74 | 151 (-3.2) | 144 (-2.8) | 143 (-2.7) | 151 (-3.2) |
| 9/30 run-rate, S-1/A (11/10) | 71 | 97 (-3.2) | 92 (-2.5) | 92 (-2.4) | 97 (-3.2) |
| 10/31 run-rate, S-1/A (11/10) | 81 | 120 (-3.6) | 111 (-2.5) | 109 (-2.4) | 119 (-3.5) |

The six S-1 revenue numbers of 10/26 (2025 quarters, FY2025, Q1/Q2 2026) are slightly under-predicted
(z -0.04 to +0.83) under every variant. The bias is specific to H2-2026 run-rates. In the filter's own innovations
(one pass over all rows public on 11/12, effective-date order; `indicator_slope_innov_2026-11-13.csv`), no model or
variant has a Ljung-Box p below 0.05.

## 6. What the indicators imply, and what the refit actually estimates

`indicator_implied_2026-11-13.png` shows, for each reading of an indicator, the run-rate implied through the
relation the filter knew the day before the reading was published:
(log I_t - a_i - c_f) / beta_i - s_seg + b, from the online filter on the rows public then, with that date's
refit parameters. Lines: baseline vs P1. Dots: point run-rate claims. Data: `indicator_slope_2026-11-13/implied_nofil.csv`.

- **SDK (B):** from July 2026 the implied run-rate climbs from about $80B to $100-180B, while the claims stay at
  $70-81B. Median since July: baseline $111B, P1 $106B. P1's line is almost on top of the baseline's: the slope
  never learns the gap in real time.
- **OpenCode (OC):** P1 lowers the implied run-rate in H2 2026 (median $88B -> $79B).
- **OpenRouter (E) and GitHub share (G):** roughly on the claims; P1 is slightly higher than the baseline.

Refit parameters at selected vintages (no filings; full table: `indicator_slope_params_2026-11-13.csv`). Drift is
the daily sd; slope and its sd are in log per week, filtered at the vintage.

| model | variant | vintage | beta | drift | slope | slope sd | sigma_g |
|:--|:--|:--|--:|--:|--:|--:|--:|
| B (npm SDK) | baseline | 2025-10-15 | 0.67 | 0.0034 |  |  | 0.006 |
| B | baseline | 2026-05-27 | 0.83 | 0.0051 |  |  | 0.017 |
| B | baseline | 2026-11-09 | 0.91 | 0.0118 |  |  | 0.015 |
| B | P1 | 2025-10-15 | 0.71 | 0.0036 | -0.0026 | 0.0063 | 0.005 |
| B | P1 | 2026-05-27 | 0.86 | 0.0050 | -0.0009 | 0.0063 | 0.017 |
| B | P1 | 2026-09-30 | 0.87 | 0.0127 | +0.0043 | 0.0071 | 0.015 |
| B | P1 | 2026-11-09 | 0.86 | 0.0122 | +0.0101 | 0.0069 | 0.015 |
| OC | baseline | 2025-10-15 | 1.28 | 0.0100 |  |  | 0.780 |
| OC | baseline | 2026-09-30 | 1.27 | 0.0104 |  |  | 0.761 |
| OC | baseline | 2026-11-09 | 1.25 | 0.0103 |  |  | 0.010 |
| OC | P1 | 2026-05-27 | 1.29 | 0.0100 | +0.0057 | 0.0127 | 0.583 |
| OC | P1 | 2026-09-30 | 1.28 | 0.0108 | +0.0156 | 0.0114 | 0.795 |
| OC | P1 | 2026-11-09 | 1.25 | 0.0125 | +0.0210 | 0.0106 | 0.039 |
| G | P1 | 2026-11-09 | 0.60 | 0.0166 | -0.0089 | 0.0074 | 0.013 |
| R | P1 | 2026-11-09 | 0.59 | 0.0049 | -0.0104 | 0.0052 | 0.014 |

Why P1 does not fix B:
- **The premise "the relation barely adapts" was only partly right.** Beta, drift and obs_sd are refit at every
  vintage for every model indicator with public rows. The SDK drift is refit at 0.003-0.015 per sqrt(day) (0.012 by
  autumn 2026, about 4x the catalog value, i.e. about 3% per week), not 0.003. The OpenCode beta is refit at
  1.21-1.33, not 1.0. The rising SDK beta (0.67 -> 0.91, the "wrong direction") is confirmed.
- **The excess growth started only in H2 2026.** In H1 2026 SDK downloads and the claims rose together (the
  implied line runs through the claims up to April). The slope prior (sd 0.01 per week) and process noise (0.001 per
  week per week) let the slope move only about 0.007 per week over a year. By 11/09 the learned SDK slope is
  +0.010 +/- 0.007 per week, about 0.04 per month, against roughly 0.1 per month of excess growth in H2. The fixed
  values were not tuned, as pre-registered.
- **Few point claims in H2 2026 before late October.** Only Funda's 71 (7/28) and the Q2 leak (8/17) arrived.
  Weekly SDK readings with obs_sd about 0.04 dominate between them. One lower claim is itself too high: NYT
  "over $100B" at 9/30, secondary tier, read as [100, 115] under iv115. Claims-only A also over-predicts H2 2026
  at period_end (10/31: 112 vs 81), so part of the overestimate is the extrapolated H1 growth plus that claim, not
  the indicator.

## 7. What got worse

- G (P1): from mostly too low to mostly too high (period_end 3 -> 10 of 13 too high), and the errors became serially
  correlated (`-+-++++++++-+` acf1 -0.07 -> `----+++------` +0.57). Signed change -0.105 (-0.163, -0.057).
- R (P1): period_end 2 -> 9 of 13 too high, `----++++-----` (acf1 +0.49). 80% coverage under the publication rule
  falls 0.93 -> 0.67 (nofil) and 0.93 -> 0.73 (pub). Early on, beta is refit to about 0.15, so the Ramp share
  barely moves revenue.
- E (P1'): 80% coverage 0.92 -> 0.69 (period_end) and 0.87 -> 0.73 (publication), with no gain in error.
- OC (P1): signed error at period_end shifts by +0.57 because the late revenue explosions are gone, but early
  vintages still have sigma_g of 0.5-0.8. Under P1, OC's Q3 revenue forecast on 11/01 drops to 13.5 (actual 16.96;
  baseline 17.0), and its forecasts of the 2025 quarters are still far off (Q2 2025: 0.41 vs 0.787).
- P1'': slightly worse than baseline for G, R and OC (intervals above zero, at most +0.034).

## 8. Side findings (not fixed: outside P's scope, and fixing them would change committed results)

1. **OC's failure is the process-noise refit.** `prepublication_forecast` refits sigma_l, sigma_g, sigma_b and
   kappa *before* the indicator, with OpenCode at its catalog prior (beta 1, drift 0.01). The filter then explains
   OpenCode's roughly 3x faster growth as revenue-growth volatility: sigma_g is 0.5-0.8 vs about 0.015 in B, with
   kappa 1.3-4.7. The revenue accumulators explode: the FY2025 forecast at 2025-12-31 is 2777 with q10 negative.
   This dominates OC's mean error (-1.13 now; -1.55 under the old one-sided rule). P1 fixes it only at the latest
   vintages, where the slope has learned about 0.02 per week.
2. **A mild parameter look-ahead in the existing refit (B).** `prepublication_forecast(..., refit=True)` runs
   `calibrate.fit_process_noise(vint, p)` with `p` from `calibrate.load_params()`. That `p` carries params.json's
   full-sample indicator fits (npm beta 0.956, drift 0.020, ...) and full-sample sigma values as the Nelder-Mead
   start; the indicators are reset to catalog priors only afterwards. Starting from defaults instead
   (`backtest.unfitted_params()`) moves B's forecasts: rr_2026_10_s1a 120 -> 107, rr_2026_05 64.3 -> 67.1,
   rev_2026q3 unchanged (sigma_b lands at 0.0019 vs 0.0186). The direction is not consistent, so this is mostly
   optimizer start-point sensitivity, but full-sample values do enter. All variants here share it, so the
   comparisons are fair.
3. **The coordinator's lag-1 autocorrelation of 0.79 for B was not reproduced.** I get 0.43-0.59 depending on the
   order of the two targets with the 2026-09-30 vintage and on the estimator. With n = 13 the standard error of a
   lag-1 autocorrelation is about 0.28. The sign strings and the runs counts are the robust part.

## 9. Files

Changed (opt-in only; defaults reproduce the previous code bit-for-bit, tested against `git show 1ff145b`):
- `config.py`: `INDICATOR_SLOPE`, `INDICATOR_DRIFT`, `INDICATOR_BETA_SD` (all None).
- `model.py`: `Layout.slopes` / `slp()` (slope states placed before the revenue accumulators, empty by default);
  slope priors and process noise in `_context`; the 1/7 coupling in `_transition`; the drift override in
  `_context`; `Result.indicator_slope()`.
- `calibrate.py`: `fit_indicators` uses `INDICATOR_BETA_SD` as the beta prior sd and, with `INDICATOR_DRIFT`, fits
  only beta and obs_sd.

New:
- `indicator_slope.py`: variants, model catalog patches, serial-correlation helpers, post-hoc correction.
- `plot_evolution_v2.py`.
- `scripts/forecast_indicator_slope.py`, `scripts/onestep_indicator_slope.py`, `scripts/plot_indicator_implied.py`,
  `scripts/eval_indicator_slope.py`, `scripts/plot_forecast_evolution_v2.py`.
- `tests/test_indicator_slope.py`: default bit-identical to 1ff145b (model on A/B/OC/G frames, with and without
  iv115; fit_indicators); slope layout and transition; P1 learns a synthetic excess growth; P1'/P1'' reach the
  refit; P1 forecasts unchanged when rows published after the vintage are deleted and when the target's value is
  tripled (G with refit; B and OC; both rules; both evidence sets); baseline reproduces sub-task N; pub equals nofil
  where no filing row was public.
- Reports: `forecast_evolution_v2_2026-11-13.{png,csv}`, `indicator_implied_2026-11-13.png`,
  `indicator_slope_onestep_2026-11-13.png`, `indicator_slope_{summary,paired,onestep,onestep_long,innov,long,params}_2026-11-13.csv`,
  and `indicator_slope_2026-11-13/` with per-run `bt_*.csv`, `params_*.json`, `onestep_*.csv`, `innov_*.csv` and
  `implied_nofil.csv`.

Reproduce (repo root, `PYTHONPATH=.`): `forecast_indicator_slope.py <B|E|G|R|OC> <variant> <nofil|pub>
publication,period_end`, then `onestep_indicator_slope.py <model> <variant> <evidence>`, then
`plot_indicator_implied.py nofil`, `eval_indicator_slope.py` and `plot_forecast_evolution_v2.py`.
