# Model-zoo diagnostics: why AUTO loses to G and E (2026-11-12)

Data as of 2026-11-10, the same inputs as `forecast_zoo_*_2026-11-10.csv`. Script: `runrate_nowcast/scripts/zoo_diagnostics.py` (`--parts ss,d1,d3,d2,report`, 4 workers, about 9 minutes of wall time). Intermediate tables are in `reports/zoo_diagnostics_<part>_2026-11-12.csv`. No package file was changed. Every forecast is recomputed with the package's own functions (`panel_model` labels/pairs/folds, `model_zoo` members, `backtest.prepublication_forecast`).

Replication checks:

- The zoo ridge (g_spec, pooled) recomputed here equals the CSV on every post-t0 target (max |diff| 0.0e+00).
- The inner-CV losses recomputed in D2 equal the CSV's `cv_loss`.
- The state-space replica reproduces the panel's claims-only path (S1r), A (S3) and G (S5) to 3 decimals.

Errors are log(actual / forecast); negative means the forecast was too high. |err| is the absolute value.

## Summary

| Strange result | Explained? | Mechanism (section) |
|---|---|---|
| 1. AUTO 0.234 vs G 0.171 / E 0.145 on the 7 post-t0 targets | Yes | The gap comes from the framework, not from the extra features (D1). With the same single signal as G or E, the zoo ridge does worse (0.314 / 0.400) than with 8 features (0.207). The predicted weekly growth is about the training-sample mean growth of the company (0.040-0.058/week). CV-chosen shrinkage removes most of the momentum and feature terms. The elasticity has to be learned from smoothed labels that rest on 3-10 anchors. A signal that starts after t0 (OpenRouter, Nov 2025) cannot enter at all. G and E use a prior elasticity and a drifting intercept instead. |
| 2. Flat inner-CV losses; MLP picked 12/14 in the own pool | Yes | The inner CV is not a vintage test (D2). Its folds train on labels that did not exist at the block start: at the Feb-Apr vintages, 2 of 2-3 own-pool folds have no real-time labels at all. Its validation pairs are interior windows of a smoothed path, and about half of the pooled validation rows are OpenAI's. On the real task rebuilt from nested vintages, the losses at the H2 vintages are 1.2x (own) and 2.0x (pooled) the inner-CV losses. The member ranking is unrelated to the inner-CV ranking (Spearman -0.07 to +0.17). The MLP drops from rank 1 to 11 (fold-nested labels) and to 14 (real task) at 2026-11-09. More than half of its +214% at rr_2026_10 comes from late-starting features that enter far outside their training range (D3): the log error goes from -1.145 to -0.499 without them. |
| 3. +31 features: AUTO worse, ridge better | Yes | Selection noise (D3). On the pooled publication targets, AUTO's +0.054 is -0.025 from the features (the same member gets better) and +0.079 from AUTO switching between near-tied members: midas at rr_2026_05 and lasso instead of ridge at the H2 vintages. The pooled lasso zeroes every late-starting feature, so imputation does not affect it. |
| 4. Pre-t0 targets: claims-only path, e.g. rr_2025_12 +18.7% vs G +4.0% | Yes | The pre-t0 path is the claims-only smoother (D4). G's advantage there comes almost entirely from its GitHub indicator bending the path between sparse anchors (mean |err| 0.059 to 0.028). The process-noise fit contributes 0, and the crossing position, segment claims and Jensen term are each at most 0.01 on average. These 7 targets carry no information about the ML members. |

**Issues found (evidence in the sections):**
1. **Inner CV leaks hindsight labels (D2).** It is the source of the flat losses and of the MLP picks.
2. **Scaling of sparse features (D3).** `Prep` scales each column by its RMS over all training rows, including the zeros imputed for missing rows. A feature present in 3% of rows (vercel_spend_share) enters the forecast row at z = -88 (own) / -160 (pooled), against a training 99th percentile of 29 / 49. Nothing is clipped, so the nonlinear members (MLP) extrapolate. This is a design flaw, not a crash.
3. **The crossing position differs between the zoo and the references.** The panel and zoo date crossing claims at rho = 0.5 (`params.json` has no `crossing_position`). A, E and G use the profiled rho of 0.60-0.92. For the crossing targets rr_2026_04 and rr_2026_05, the zoo and G therefore score the same official number at dates 8-10 days apart (table D1c). The paired comparisons are slightly apples-to-oranges. The effect on the gap is mixed and small.
4. **cumsum of carried statements.** The `all` rule turns `cloud_bedrock_growth` (a quarterly growth statement carried for 100 days) into a weekly cumsum level. Interior gaps (12 weeks) are zero-filled, so a stale statement reads as a slowdown. npm series zero-fill 3 holiday weeks. Lasso zeroes this feature; it is a construction bug of small consequence.
5. **Minor: the revenue label path uses the median, not the mean.** `path_value` integrates exp(mean log level), while the smoother's own revenue is the mean. Pre-t0 revenue forecasts are up to 1.1% lower than `SS claims only` (S0 vs S1).

**Recommended fixes, in priority order:**
1. **Score only post-t0 targets for model comparison.** Report pre-t0 targets separately as a label-path check: they are identical for all 18 members and dilute every paired test (7 of 14 publication targets).
2. **Use vintage-nested evaluation and stop selecting per vintage among 18 members.** Nested CV removes the MLP's spurious advantage. Its own picks were no better on the test targets, because 3-14 real-task points per vintage cannot rank 18 near-equivalent members. Fix one or two simple members in advance (ridge, Bayesian ridge), or use an equal-weight average, and use nested CV only for shrinkage.
3. **Change the framework's anchor: ML corrects the state-space path, not y(t0) + g x mean growth.** Use the claims-only (or G/E) state-space forecast as the base and let the ML predict the deviation of growth from it. Or put a prior elasticity on each feature (as the state space does) instead of learning it from 3-10 anchors. Replace the company-mean intercept, which carries 2025-26 hyper-growth into every forecast, with the current state-space growth.
4. **Standardize on present rows and clip out-of-range inputs**, or drop features with less than about 6 months of overlap with the training labels. Do not zero-fill interior gaps of cumsum features. Restrict nonlinear members to features with full history, or drop them.
5. **Align the crossing position** between the zoo and the references (profile rho at each vintage, as A, E and G do), so that paired comparisons score the same target date.

## D1. Framework vs features

This runs the zoo pipeline (`build` reproduces `model_zoo.forecast_target`: same labels, pairs, CV folds, ridge grid and forecast path) with the pooled pool and ridge only. The feature sets are:

- `gh`: github_repo_share only, the signal of G.
- `or_spend`: openrouter_log_tokens + log_price, the tokens x price of E. OpenAI rows use tokens only, since price is Anthropic-only.
- `or_tp`: the same two columns separately.

Each is also run re-dated at G's profiled crossing position (`@rhoG`; claims and target t_eff re-dated). `gh alpha=0.01` is a mechanism probe with the shrinkage fixed near zero; it is not a candidate model. References are AUTO (g_spec, pooled) and the state-space A / G / E from the CSVs. `SS panel` is the claims-only smoother from the same label run.

### D1a. Publication rule, the 7 post-t0 targets

|                  |   AUTO |   ridge g_spec |   ridge gh |   ridge or_spend |   ridge or_tp |   ridge gh@rhoG |   ridge g_spec@rhoG |   ridge gh alpha=0.01 |   SS panel |   SS panel@rhoG |      A |      G |      E |
|:-----------------|-------:|---------------:|-----------:|-----------------:|--------------:|----------------:|--------------------:|----------------------:|-----------:|----------------:|-------:|-------:|-------:|
| rev_2026q3       | -0.041 |         -0.041 |     -0.162 |           -0.293 |        -0.280 |          -0.240 |              -0.130 |                -0.123 |     -0.237 |          -0.319 | -0.324 |  0.014 |  0.051 |
| rr_2026_02       |  0.359 |          0.382 |      0.393 |            0.365 |         0.365 |           0.385 |               0.364 |                 0.393 |      0.374 |           0.363 |  0.363 |  0.113 |  0.096 |
| rr_2026_03       |  0.238 |          0.103 |      0.241 |            0.241 |         0.045 |           0.235 |               0.163 |                 0.242 |      0.242 |           0.236 |  0.217 |  0.180 |  0.087 |
| rr_2026_04       |  0.379 |          0.415 |      0.379 |            0.389 |         0.451 |           0.301 |               0.311 |                 0.294 |      0.207 |           0.123 |  0.108 |  0.197 |  0.108 |
| rr_2026_05       |  0.128 |          0.015 |      0.104 |            0.127 |         0.202 |           0.096 |               0.020 |                -0.091 |     -0.310 |          -0.228 | -0.206 |  0.098 |  0.120 |
| rr_2026_09_s1a   | -0.233 |         -0.233 |     -0.417 |           -0.616 |        -0.601 |          -0.455 |              -0.283 |                -0.357 |     -0.545 |          -0.568 | -0.568 | -0.291 |  0.262 |
| rr_2026_10_s1a   | -0.256 |         -0.256 |     -0.505 |           -0.770 |        -0.743 |          -0.557 |              -0.305 |                -0.437 |     -0.605 |          -0.640 | -0.642 | -0.302 | -0.288 |
| mean |log error| |  0.234 |          0.207 |      0.314 |            0.400 |         0.384 |           0.324 |               0.225 |                 0.277 |      0.360 |           0.354 |  0.347 |  0.171 |  0.145 |

### D1b. Period-end rule, the 11 post-t0 targets

|                  |   AUTO |   ridge g_spec |   ridge gh |   ridge or_spend |   ridge or_tp |   ridge gh@rhoG |   ridge g_spec@rhoG |   ridge gh alpha=0.01 |   SS panel |   SS panel@rhoG |      A |      G |      E |
|:-----------------|-------:|---------------:|-----------:|-----------------:|--------------:|----------------:|--------------------:|----------------------:|-----------:|----------------:|-------:|-------:|-------:|
| rev_2025fy       | -0.017 |         -0.020 |     -0.016 |           -0.017 |        -0.017 |           0.001 |              -0.004 |                -0.016 |     -0.050 |          -0.034 | -0.034 |  0.127 | -0.040 |
| rev_2026q1       |  0.088 |          0.092 |      0.088 |            0.092 |         0.118 |           0.082 |               0.088 |                 0.044 |     -0.068 |          -0.026 | -0.021 |  0.118 | -0.040 |
| rev_2026q2       | -0.215 |         -0.215 |     -0.165 |           -0.224 |        -0.238 |          -0.074 |              -0.092 |                -0.111 |     -0.247 |          -0.120 | -0.120 |  0.558 | -0.127 |
| rev_2026q3       | -0.267 |         -0.289 |     -0.310 |           -0.318 |        -0.305 |          -0.363 |              -0.354 |                -0.125 |     -0.237 |          -0.319 | -0.324 | -0.012 |  0.234 |
| rr_2025_12       |  0.065 |          0.055 |      0.069 |            0.065 |         0.065 |           0.070 |               0.052 |                 0.069 |      0.043 |           0.040 |  0.040 |  0.101 | -0.003 |
| rr_2026_02       |  0.359 |          0.382 |      0.393 |            0.365 |         0.365 |           0.385 |               0.364 |                 0.393 |      0.374 |           0.363 |  0.363 |  0.113 |  0.097 |
| rr_2026_03       |  0.238 |          0.103 |      0.241 |            0.241 |         0.045 |           0.229 |               0.157 |                 0.242 |      0.242 |           0.231 |  0.217 |  0.211 |  0.141 |
| rr_2026_04       |  0.379 |          0.400 |      0.395 |            0.396 |         0.402 |           0.308 |               0.337 |                 0.315 |      0.207 |           0.124 |  0.108 |  0.192 |  0.081 |
| rr_2026_05       |  0.128 |          0.130 |      0.143 |            0.129 |         0.203 |           0.096 |               0.091 |                -0.088 |     -0.310 |          -0.228 | -0.206 |  0.098 |  0.171 |
| rr_2026_09_s1a   | -0.526 |         -0.566 |     -0.610 |           -0.623 |        -0.607 |          -0.618 |              -0.591 |                -0.313 |     -0.416 |          -0.490 | -0.494 | -0.124 |  0.376 |
| rr_2026_10_s1a   | -0.184 |         -0.184 |     -0.459 |           -0.739 |        -0.711 |          -0.524 |              -0.261 |                -0.386 |     -0.419 |          -0.529 | -0.537 | -0.005 | -0.401 |
| mean |log error| |  0.224 |          0.221 |      0.263 |            0.292 |         0.280 |           0.250 |               0.217 |                 0.191 |      0.238 |           0.228 |  0.224 |  0.151 |  0.156 |

### D1c. What the ridge forecast is made of (publication, `gh`)

Predicted weekly log growth after t0 = company mean (`rate_base`, the weighted mean of the training labels' growth) + momentum term + feature term, averaged over the target's dates after t0. `rate_needed` is the constant weekly growth from y(t0) that would hit the actual value. `alpha` is the CV-chosen ridge penalty. `elast` is in original units (weekly growth per unit of weekly feature change).

|                                     | t0         |   g_weeks |     alpha |   rate_base |   rate_mom |   rate_features |   rate_pred |   rate_needed |   mom_input |   elast:mom |   share chg/wk |   elast share |
|:------------------------------------|:-----------|----------:|----------:|------------:|-----------:|----------------:|------------:|--------------:|------------:|------------:|---------------:|--------------:|
| ('rev_2026q3', 'gh')                | 2026-07-14 |    5.6429 |  100.0000 |      0.0585 |     0.0003 |         -0.0294 |      0.0295 |      nan      |      0.0445 |      0.3192 |        -0.0085 |        1.4592 |
| ('rev_2026q3', 'gh alpha=0.01')     | 2026-07-14 |    5.6429 |    0.0100 |      0.0585 |     0.0004 |         -0.0370 |      0.0220 |      nan      |      0.0445 |      0.4220 |        -0.0085 |        1.8376 |
| ('rr_2026_02', 'gh')                | 2025-10-10 |   15.4286 |    0.0100 |      0.0396 |    -0.0189 |         -0.0018 |      0.0190 |        0.0445 |      0.0298 |      0.7754 |         0.0191 |       -0.1170 |
| ('rr_2026_02', 'gh alpha=0.01')     | 2025-10-10 |   15.4286 |    0.0100 |      0.0396 |    -0.0189 |         -0.0018 |      0.0190 |        0.0445 |      0.0298 |      0.7754 |         0.0191 |       -0.1170 |
| ('rr_2026_03', 'gh')                | 2026-01-26 |    1.8571 |  100.0000 |      0.0410 |    -0.0005 |         -0.0003 |      0.0402 |        0.1700 |      0.0420 |      0.1900 |         0.0000 |        0.0545 |
| ('rr_2026_03', 'gh alpha=0.01')     | 2026-01-26 |    1.8571 |    0.0100 |      0.0410 |    -0.0009 |         -0.0007 |      0.0395 |        0.1700 |      0.0420 |      0.3005 |         0.0000 |        0.1383 |
| ('rr_2026_04', 'gh')                | 2026-02-08 |    3.7143 | 1000.0000 |      0.0414 |     0.0023 |          0.0026 |      0.0462 |        0.1481 |      0.0922 |      0.0398 |         0.0187 |        0.2455 |
| ('rr_2026_04', 'gh alpha=0.01')     | 2026-02-08 |    3.7143 |    0.0100 |      0.0414 |     0.0152 |          0.0123 |      0.0688 |        0.1481 |      0.0922 |      0.2680 |         0.0187 |        1.1800 |
| ('rr_2026_05', 'gh')                | 2026-03-06 |    8.0000 | 1000.0000 |      0.0491 |     0.0038 |          0.0017 |      0.0546 |        0.0676 |      0.1077 |      0.0522 |         0.0154 |        0.3942 |
| ('rr_2026_05', 'gh alpha=0.01')     | 2026-03-06 |    8.0000 |    0.0100 |      0.0491 |     0.0226 |          0.0073 |      0.0790 |        0.0676 |      0.1077 |      0.3064 |         0.0154 |        1.7295 |
| ('rr_2026_09_s1a', 'gh')            | 2026-07-14 |    9.1429 |  100.0000 |      0.0584 |     0.0021 |         -0.0276 |      0.0329 |       -0.0126 |      0.0500 |      0.3248 |        -0.0063 |        1.4376 |
| ('rr_2026_09_s1a', 'gh alpha=0.01') | 2026-07-14 |    9.1429 |    0.0100 |      0.0584 |     0.0028 |         -0.0348 |      0.0264 |       -0.0126 |      0.0500 |      0.4291 |        -0.0063 |        1.8115 |
| ('rr_2026_10_s1a', 'gh')            | 2026-07-14 |   13.5714 |  100.0000 |      0.0584 |     0.0032 |         -0.0232 |      0.0384 |        0.0012 |      0.0534 |      0.3248 |        -0.0032 |        1.4376 |
| ('rr_2026_10_s1a', 'gh alpha=0.01') | 2026-07-14 |   13.5714 |    0.0100 |      0.0584 |     0.0042 |         -0.0292 |      0.0334 |        0.0012 |      0.0534 |      0.4291 |        -0.0032 |        1.8115 |

Crossing positions and target dates (rho = 0.5 in the zoo):

|                                   | vintage    |   rho_A |   rho_G |   rho_E | t_eff_05   | t_eff_rhoG   |
|:----------------------------------|:-----------|--------:|--------:|--------:|:-----------|:-------------|
| ('publication', 'rr_2026_02')     | 2026-02-11 |    0.73 |    0.73 |    0.72 | 2026-01-26 | 2026-01-26   |
| ('publication', 'rr_2026_03')     | 2026-03-02 |    0.73 |    0.60 |    0.67 | 2026-02-08 | 2026-02-09   |
| ('publication', 'rr_2026_04')     | 2026-04-05 |    0.88 |    0.80 |    0.82 | 2026-03-06 | 2026-03-16   |
| ('publication', 'rr_2026_05')     | 2026-05-27 |    0.91 |    0.80 |    0.87 | 2026-05-01 | 2026-05-09   |
| ('publication', 'rr_2026_09_s1a') | 2026-11-09 |    0.92 |    0.89 |    0.89 | 2026-09-16 | 2026-09-16   |
| ('publication', 'rev_2026q3')     | 2026-11-01 |    0.92 |    0.89 |    0.89 | 2026-09-30 | 2026-09-30   |
| ('publication', 'rr_2026_10_s1a') | 2026-11-09 |    0.92 |    0.89 |    0.89 | 2026-10-17 | 2026-10-17   |
| ('period_end', 'rr_2025_12')      | 2025-12-28 |    0.73 |    0.73 |    0.72 | 2025-12-14 | 2025-12-14   |
| ('period_end', 'rev_2025fy')      | 2025-12-31 |    0.73 |    0.73 |    0.72 | 2025-12-31 | 2025-12-31   |
| ('period_end', 'rr_2026_02')      | 2026-02-09 |    0.73 |    0.73 |    0.72 | 2026-01-26 | 2026-01-26   |
| ('period_end', 'rr_2026_03')      | 2026-02-22 |    0.73 |    0.61 |    0.65 | 2026-02-08 | 2026-02-10   |
| ('period_end', 'rr_2026_04')      | 2026-03-20 |    0.88 |    0.82 |    0.82 | 2026-03-06 | 2026-03-17   |
| ('period_end', 'rev_2026q1')      | 2026-03-31 |    0.88 |    0.82 |    0.82 | 2026-03-31 | 2026-03-31   |
| ('period_end', 'rr_2026_05')      | 2026-05-15 |    0.91 |    0.80 |    0.87 | 2026-05-01 | 2026-05-09   |
| ('period_end', 'rev_2026q2')      | 2026-06-30 |    0.90 |    0.90 |    0.89 | 2026-06-30 | 2026-06-30   |
| ('period_end', 'rr_2026_09_s1a')  | 2026-09-30 |    0.92 |    0.89 |    0.89 | 2026-09-16 | 2026-09-16   |
| ('period_end', 'rev_2026q3')      | 2026-09-30 |    0.92 |    0.89 |    0.89 | 2026-09-30 | 2026-09-30   |
| ('period_end', 'rr_2026_10_s1a')  | 2026-10-31 |    0.92 |    0.89 |    0.89 | 2026-10-17 | 2026-10-17   |

Coverage of the single signals in the training pairs (publication):

|                                         | vintage    |   anth_pairs |   anth_pairs_with_signal |   openai_pairs_with_signal | signal_in_forecast_window   | first_signal_week   |
|:----------------------------------------|:-----------|-------------:|-------------------------:|---------------------------:|:----------------------------|:--------------------|
| ('rr_2026_02', 'github_repo_share')     | 2026-02-11 |           73 |                       73 |                         59 | True                        | 2025-01-30          |
| ('rr_2026_02', 'or_spend')              | 2026-02-11 |           73 |                        0 |                          0 | False                       | 2025-11-20          |
| ('rr_2026_03', 'github_repo_share')     | 2026-03-02 |          118 |                      118 |                         59 | True                        | 2025-01-30          |
| ('rr_2026_03', 'or_spend')              | 2026-03-02 |          118 |                        8 |                          0 | True                        | 2025-11-20          |
| ('rr_2026_04', 'github_repo_share')     | 2026-04-05 |          124 |                      124 |                        152 | True                        | 2025-01-30          |
| ('rr_2026_04', 'or_spend')              | 2026-04-05 |          124 |                       12 |                         26 | True                        | 2025-11-20          |
| ('rr_2026_05', 'github_repo_share')     | 2026-05-27 |          136 |                      136 |                        158 | True                        | 2025-01-30          |
| ('rr_2026_05', 'or_spend')              | 2026-05-27 |          136 |                       23 |                         32 | True                        | 2025-11-20          |
| ('rr_2026_09_s1a', 'github_repo_share') | 2026-11-09 |          190 |                      190 |                        233 | True                        | 2025-01-30          |
| ('rr_2026_09_s1a', 'or_spend')          | 2026-11-09 |          190 |                       77 |                        107 | True                        | 2025-11-20          |
| ('rev_2026q3', 'github_repo_share')     | 2026-11-01 |          190 |                      190 |                        233 | False                       | 2025-01-30          |
| ('rev_2026q3', 'or_spend')              | 2026-11-01 |          190 |                       77 |                        107 | False                       | 2025-11-20          |
| ('rr_2026_10_s1a', 'github_repo_share') | 2026-11-09 |          190 |                      190 |                        233 | True                        | 2025-01-30          |
| ('rr_2026_10_s1a', 'or_spend')          | 2026-11-09 |          190 |                       77 |                        107 | True                        | 2025-11-20          |

**Conclusion D1:** the gap comes from the framework, not from the extra features.

- **Fewer features make it worse.** On the same 7 targets, the zoo ridge with G's signal alone scores 0.314 and with E's signal alone 0.400, against G 0.171 and E 0.145. That is worse than the 8-feature ridge (0.207) and near the claims-only smoother (0.360). The period-end rule shows the same pattern.
- **What the framework does.** It extrapolates y(t0) at the company's mean training growth (0.040-0.058/week, about +20-29% a month). At most vintages CV picks alpha 100-1000, which shrinks the momentum elasticity to 0.04-0.3 and the feature terms to about 0.002/week (5 of 7 targets). The early-2026 targets needed 0.045-0.17/week against a predicted 0.02-0.05 (forecasts too low: rr_2026_02/03/04). The H2-2026 targets needed about 0 (forecasts too high: rr_2026_09/10).
- **The GitHub signal points the right way but is too weak.** In H2 the feature term is -0.023 to -0.029/week against the -0.06 needed. At rr_2026_02 the learned elasticity even has the wrong sign (-0.12), learned from the 2025 pairs.
- **The OpenRouter signal cannot enter.** It starts on 2025-11-20 (rolling 53-week history). It has 0 Anthropic training pairs at the rr_2026_02 vintage, where it is also missing at t0, and 8-23 pairs up to May 2026. Its elasticity is about 0.
- **How the state space differs.** It uses the same signal with a prior elasticity (beta 1 +/- 1) and a drifting intercept, so it needs no labelled history for the signal.
- **Shrinkage is not the main problem.** Removing it (`alpha=0.01`) helps only partly (gh 0.277 publication, 0.191 period-end).
- **Re-dating crossing claims does not close the gap** (`gh@rhoG` 0.324). It mainly moves rr_2026_04 and rr_2026_05. Once re-dated, the panel's claims-only smoother matches A (`SS panel@rhoG` 0.354 vs A 0.347 publication; 0.228 vs 0.224 period-end). The claims side of the two pipelines therefore agrees apart from the crossing position.
- **Where the gap sits.** The early-2026 hyper-growth targets (rr_2026_02/03/04) account for most of it; there G and E saw the acceleration through their indicators. At the H2 targets, G at the publication vintage is no better than the zoo ridge (-0.29 / -0.30 vs -0.23 / -0.26).

## D2. Validity of the inner CV

For each publication vintage of a post-t0 target and each pool, with the g_spec features and every member at the hyperparameters its inner CV chose:

- **inner CV:** `pm.cv_folds` on the pairs built from the vintage-v labels (as in the zoo).
- **fold-nested labels:** the same validation pairs, but the training labels, pairs and features are rebuilt from the claims and features public the day before the block starts (`company_labels(c, claims, b0 - 1 day)`). A fold is usable when that gives at least 20 pairs.
- **nested real task:** origins are the publication dates of the Anthropic anchors (one per distinct t0, at least 3 anchors). At each origin u the member is trained on the pool's pairs at u and forecasts y(t0_u + h) for h = 4, 8, 13 weeks, as the zoo does: from y_u(t0_u), with momentum from the u-path and feature changes over (t0_u, d]. The truth is the vintage-v label (only dates <= t0_v). Loss = mean |level error| / h, the same units as the CV loss.

`test |err|` is the member's mean |log error| on the publication targets at that vintage, from the zoo CSV.

|                                        | ('2026-11-09', 'pooled')       | ('2026-11-09', 'own')          | ('2026-11-01', 'pooled')   | ('2026-11-01', 'own')   | ('2026-05-27', 'pooled')   | ('2026-05-27', 'own')   | ('2026-04-05', 'pooled')   | ('2026-04-05', 'own')   | ('2026-03-02', 'pooled')   | ('2026-03-02', 'own')   | ('2026-02-11', 'pooled')   | ('2026-02-11', 'own')   |
|:---------------------------------------|:-------------------------------|:-------------------------------|:---------------------------|:------------------------|:---------------------------|:------------------------|:---------------------------|:------------------------|:---------------------------|:------------------------|:---------------------------|:------------------------|
| targets                                | rr_2026_09_s1a, rr_2026_10_s1a | rr_2026_09_s1a, rr_2026_10_s1a | rev_2026q3                 | rev_2026q3              | rr_2026_05                 | rr_2026_05              | rr_2026_04                 | rr_2026_04              | rr_2026_03                 | rr_2026_03              | rr_2026_02                 | rr_2026_02              |
| n_folds                                | 4                              | 3                              | 4                          | 3                       | 3                          | 3                       | 3                          | 3                       | 3                          | 3                       | 3                          | 2                       |
| n_folds_nested_usable                  | 3                              | 2                              | 3                          | 2                       | 3                          | 1                       | 3                          | 1                       | 2                          | 1                       | 2                          | 0                       |
| inner_min                              | 0.0142                         | 0.0207                         | 0.0143                     | 0.0210                  | 0.0158                     | 0.0324                  | 0.0133                     | 0.0249                  | 0.0035                     | 0.0050                  | 0.0059                     | 0.0054                  |
| inner_max                              | 0.0169                         | 0.0599                         | 0.0171                     | 0.0619                  | 0.0219                     | 0.0511                  | 0.0198                     | 0.0379                  | 0.0082                     | 0.0080                  | 0.0147                     | 0.0214                  |
| inner_spread_rel                       | 0.1693                         | 1.5251                         | 0.1754                     | 1.5572                  | 0.3394                     | 0.5009                  | 0.4200                     | 0.4813                  | 0.7009                     | 0.4281                  | 1.0845                     | 1.5601                  |
| n_within_10pct_of_best_inner           | 6                              | 1                              | 7                          | 1                       | 4                          | 3                       | 6                          | 10                      | 1                          | 2                       | 2                          | 2                       |
| n_oos_points                           | 14                             | 14                             | 14                         | 14                      | 4                          | 4                       | 3                          | 3                       | 3                          | 3                       | 0                          | 0                       |
| oos_spread_rel                         | 0.5519                         | 0.8588                         | 0.5338                     | 0.8314                  | 0.3833                     | 0.2490                  | 0.5024                     | 0.5516                  | 1.1327                     | 1.3974                  |                            |                         |
| spearman_inner_vs_oos                  | 0.0609                         | 0.1662                         | -0.0712                    | 0.1517                  | -0.6450                    | -0.2157                 | -0.5562                    | -0.1889                 | 0.4469                     | -0.6367                 |                            |                         |
| spearman_inner_vs_nestedlabels         | 0.8204                         | 0.1950                         | 0.8184                     | 0.2136                  | -0.1868                    | 0.5851                  | -0.3416                    | 0.7936                  | 0.8060                     | 0.8349                  | 0.0010                     |                         |
| mlp_rank_inner                         | 13                             | 1                              | 5                          | 1                       | 6                          | 1                       | 11                         | 8                       | 10                         | 1                       | 16                         | 4                       |
| mlp_rank_nestedlabels                  | 5                              | 11                             | 6                          | 11                      | 7                          | 2                       | 18                         | 6                       | 2                          | 10                      | 1                          |                         |
| mlp_rank_oos                           | 17                             | 14                             | 17                         | 14                      | 5                          | 4                       | 6                          | 8                       | 9                          | 10                      |                            |                         |
| pick: inner CV (AUTO)                  | ridge                          | mlp                            | ridge                      | mlp                     | kernel_ridge               | mlp                     | rf                         | pca                     | rf                         | mlp                     | pca                        | trend                   |
| test |err| of pick: inner CV (AUTO)    | 0.2445                         | 0.9372                         | 0.0411                     | 0.4077                  | 0.1284                     | 0.0394                  | 0.3790                     | 0.4585                  | 0.2383                     | 0.2139                  | 0.3595                     | 0.3602                  |
| pick: fold-nested labels               | lasso                          | xgboost                        | lasso                      | xgboost                 | lasso                      | kernel_ridge            | lasso                      | gp                      | midas                      | midas                   | mlp                        |                         |
| test |err| of pick: fold-nested labels | 0.5204                         | 0.5934                         | 0.2166                     | 0.2989                  | 0.0572                     | 0.0915                  | 0.4627                     | 0.3964                  | 0.0901                     | 0.2368                  | 0.2411                     |                         |
| pick: nested real task                 | midas                          | midas                          | midas                      | midas                   | bayes_ridge                | bayes_ridge             | enet                       | ard                     | kernel_ridge               | midas                   |                            |                         |
| test |err| of pick: nested real task   | 0.6268                         | 0.6950                         | 0.2748                     | 0.3049                  | 0.4293                     | 0.7030                  | 0.4716                     | 0.3645                  | 0.2394                     | 0.2368                  |                            |                         |
| test |err| best member (hindsight)     | 0.1135                         | 0.2563                         | 0.0201                     | 0.0850                  | 0.0081                     | 0.0140                  | 0.2636                     | 0.2814                  | 0.0033                     | 0.0196                  | 0.0644                     | 0.0312                  |

Folds: training pairs available to the inner CV vs those that existed at the block start. `anth_anchors_at_u` is the number of Anthropic anchors public at the block start. `mean_abs_rev_train_y` is the mean |y_v - y_u| on the shared training pairs.

|                             | block_start   |   n_train_inner |   n_valid |   n_train_nested |   anth_anchors_at_u |   anth_anchors_at_v |   mean_abs_rev_train_y |   valid_share_anthropic | nested_usable   |
|:----------------------------|:--------------|----------------:|----------:|-----------------:|--------------------:|--------------------:|-----------------------:|------------------------:|:----------------|
| ('2026-11-09', 'pooled', 0) | 2025-05-08    |              41 |       102 |                0 |                   0 |                  10 |               nan      |                  0.5000 | False           |
| ('2026-11-09', 'pooled', 1) | 2025-09-04    |             135 |       102 |               68 |                   2 |                  10 |                 0.0015 |                  0.5000 | True            |
| ('2026-11-09', 'pooled', 2) | 2026-01-01    |             237 |       100 |              144 |                   3 |                  10 |                 0.0036 |                  0.4900 | True            |
| ('2026-11-09', 'pooled', 3) | 2026-04-30    |             339 |        48 |              306 |                   7 |                  10 |                 0.0013 |                  0.2083 | True            |
| ('2026-11-09', 'own', 0)    | 2025-08-28    |              52 |        42 |                0 |                   1 |                  10 |               nan      |                  1.0000 | False           |
| ('2026-11-09', 'own', 1)    | 2025-12-04    |              94 |        42 |               73 |                   3 |                  10 |                 0.0052 |                  1.0000 | True            |
| ('2026-11-09', 'own', 2)    | 2026-03-12    |             136 |        29 |              124 |                   6 |                  10 |                 0.0004 |                  1.0000 | True            |
| ('2026-11-01', 'pooled', 0) | 2025-05-08    |              41 |       102 |                0 |                   0 |                  10 |               nan      |                  0.5000 | False           |
| ('2026-11-01', 'pooled', 1) | 2025-09-04    |             135 |       102 |               68 |                   2 |                  10 |                 0.0014 |                  0.5000 | True            |
| ('2026-11-01', 'pooled', 2) | 2026-01-01    |             237 |       100 |              144 |                   3 |                  10 |                 0.0037 |                  0.4900 | True            |
| ('2026-11-01', 'pooled', 3) | 2026-04-30    |             339 |        48 |              306 |                   7 |                  10 |                 0.0012 |                  0.2083 | True            |
| ('2026-11-01', 'own', 0)    | 2025-08-28    |              52 |        42 |                0 |                   1 |                  10 |               nan      |                  1.0000 | False           |
| ('2026-11-01', 'own', 1)    | 2025-12-04    |              94 |        42 |               73 |                   3 |                  10 |                 0.0055 |                  1.0000 | True            |
| ('2026-11-01', 'own', 2)    | 2026-03-12    |             136 |        29 |              124 |                   6 |                  10 |                 0.0007 |                  1.0000 | True            |
| ('2026-05-27', 'pooled', 0) | 2025-06-26    |              76 |        72 |               41 |                   1 |                   7 |                 0.0007 |                  0.5000 | True            |
| ('2026-05-27', 'pooled', 1) | 2025-09-18    |             147 |        72 |               68 |                   2 |                   7 |                 0.0014 |                  0.5000 | True            |
| ('2026-05-27', 'pooled', 2) | 2025-12-11    |             219 |        37 |              144 |                   3 |                   7 |                 0.0037 |                  0.3784 | True            |
| ('2026-05-27', 'own', 0)    | 2025-07-03    |              28 |        33 |                0 |                   1 |                   7 |               nan      |                  1.0000 | False           |
| ('2026-05-27', 'own', 1)    | 2025-09-18    |              61 |        33 |               12 |                   2 |                   7 |                 0.0013 |                  1.0000 | False           |
| ('2026-05-27', 'own', 2)    | 2025-12-04    |              94 |        17 |               73 |                   3 |                   7 |                 0.0061 |                  1.0000 | True            |
| ('2026-04-05', 'pooled', 0) | 2025-06-19    |              71 |        72 |               41 |                   1 |                   6 |                 0.0008 |                  0.5000 | True            |
| ('2026-04-05', 'pooled', 1) | 2025-09-11    |             141 |        69 |               68 |                   2 |                   6 |                 0.0014 |                  0.4783 | True            |
| ('2026-04-05', 'pooled', 2) | 2025-12-04    |             213 |        28 |              144 |                   3 |                   6 |                 0.0033 |                  0.2857 | True            |
| ('2026-04-05', 'own', 0)    | 2025-06-19    |              24 |        30 |                0 |                   1 |                   6 |               nan      |                  1.0000 | False           |
| ('2026-04-05', 'own', 1)    | 2025-08-28    |              52 |        30 |                0 |                   1 |                   6 |               nan      |                  1.0000 | False           |
| ('2026-04-05', 'own', 2)    | 2025-11-06    |              82 |        17 |               73 |                   3 |                   6 |                 0.0052 |                  1.0000 | True            |
| ('2026-03-02', 'pooled', 0) | 2025-06-05    |              61 |        38 |                0 |                   1 |                   4 |               nan      |                  0.7895 | False           |
| ('2026-03-02', 'pooled', 1) | 2025-08-14    |             117 |        30 |               56 |                   1 |                   4 |                 0.0014 |                  1.0000 | True            |
| ('2026-03-02', 'pooled', 2) | 2025-10-23    |             147 |        17 |              144 |                   3 |                   4 |                 0.0019 |                  1.0000 | True            |
| ('2026-03-02', 'own', 0)    | 2025-06-19    |              24 |        30 |                0 |                   1 |                   4 |               nan      |                  1.0000 | False           |
| ('2026-03-02', 'own', 1)    | 2025-08-28    |              52 |        27 |                0 |                   1 |                   4 |               nan      |                  1.0000 | False           |
| ('2026-03-02', 'own', 2)    | 2025-10-30    |              79 |        14 |               73 |                   3 |                   4 |                 0.0024 |                  1.0000 | True            |
| ('2026-02-11', 'pooled', 0) | 2025-04-24    |              31 |        38 |                0 |                   0 |                   3 |               nan      |                  0.5526 | False           |
| ('2026-02-11', 'pooled', 1) | 2025-06-12    |              66 |        25 |               41 |                   1 |                   3 |                 0.0010 |                  0.7600 | True            |
| ('2026-02-11', 'pooled', 2) | 2025-07-31    |             105 |        10 |               56 |                   1 |                   3 |                 0.0017 |                  1.0000 | True            |
| ('2026-02-11', 'own', 0)    | 2025-06-26    |              26 |        15 |                0 |                   1 |                   3 |               nan      |                  1.0000 | False           |
| ('2026-02-11', 'own', 1)    | 2025-08-07    |              43 |         8 |                0 |                   1 |                   3 |               nan      |                  1.0000 | False           |

Per member at 2026-11-09, own (sorted by inner-CV loss):

| member       |   cv_inner |   cv_inner_anthropic_rows |   inner_on_nested_folds |   fold_nested_labels |   oos_real_task_rate |   oos_real_task_level |   test_abs_log_err |
|:-------------|-----------:|--------------------------:|------------------------:|---------------------:|---------------------:|----------------------:|-------------------:|
| mlp          |     0.0207 |                    0.0239 |                  0.0227 |               0.0345 |               0.0357 |                0.2751 |             0.9372 |
| kernel_ridge |     0.0234 |                    0.0275 |                  0.0285 |               0.0310 |               0.0338 |                0.2512 |             0.7132 |
| extra_trees  |     0.0240 |                    0.0303 |                  0.0275 |               0.0316 |               0.0306 |                0.2220 |             0.7117 |
| ridge        |     0.0241 |                    0.0302 |                  0.0277 |               0.0363 |               0.0312 |                0.2314 |             0.6409 |
| pca          |     0.0245 |                    0.0307 |                  0.0285 |               0.0355 |               0.0304 |                0.2247 |             0.7956 |
| rf           |     0.0247 |                    0.0304 |                  0.0285 |               0.0308 |               0.0312 |                0.2241 |             0.6759 |
| gp           |     0.0249 |                    0.0297 |                  0.0297 |               0.0309 |               0.0329 |                0.2478 |             0.7132 |
| knn          |     0.0249 |                    0.0306 |                  0.0290 |               0.0380 |               0.0324 |                0.2391 |             0.8392 |
| lightgbm     |     0.0253 |                    0.0303 |                  0.0300 |               0.0262 |               0.0302 |                0.2138 |             0.5555 |
| trend        |     0.0262 |                    0.0306 |                  0.0310 |               0.0321 |               0.0322 |                0.2386 |             0.7169 |
| xgboost      |     0.0264 |                    0.0309 |                  0.0316 |               0.0252 |               0.0300 |                0.2125 |             0.5934 |
| hgb          |     0.0268 |                    0.0305 |                  0.0320 |               0.0254 |               0.0301 |                0.2149 |             0.5637 |
| lasso        |     0.0277 |                    0.0348 |                  0.0282 |               0.0301 |               0.0297 |                0.2223 |             0.6347 |
| midas        |     0.0283 |                    0.0317 |                  0.0337 |               0.0406 |               0.0273 |                0.2028 |             0.6950 |
| pls          |     0.0341 |                    0.0396 |                  0.0344 |               0.0545 |               0.0486 |                0.3780 |             0.5083 |
| enet         |     0.0449 |                    0.0524 |                  0.0489 |               0.0316 |               0.0409 |                0.3146 |             0.5260 |
| bayes_ridge  |     0.0566 |                    0.0582 |                  0.0653 |               0.0397 |               0.0546 |                0.4236 |             0.2563 |
| ard          |     0.0599 |                    0.0628 |                  0.0694 |               0.0349 |               0.0502 |                0.3865 |             0.3305 |

Per member at 2026-11-09, pooled (sorted by inner-CV loss):

| member       |   cv_inner |   cv_inner_anthropic_rows |   inner_on_nested_folds |   fold_nested_labels |   oos_real_task_rate |   oos_real_task_level |   test_abs_log_err |
|:-------------|-----------:|--------------------------:|------------------------:|---------------------:|---------------------:|----------------------:|-------------------:|
| ridge        |     0.0142 |                    0.0295 |                  0.0145 |               0.0153 |               0.0377 |                0.2857 |             0.2445 |
| lightgbm     |     0.0150 |                    0.0262 |                  0.0187 |               0.0192 |               0.0269 |                0.2088 |             0.3642 |
| pca          |     0.0151 |                    0.0305 |                  0.0158 |               0.0145 |               0.0324 |                0.2390 |             0.4649 |
| lasso        |     0.0154 |                    0.0337 |                  0.0162 |               0.0135 |               0.0307 |                0.2358 |             0.5204 |
| knn          |     0.0156 |                    0.0300 |                  0.0165 |               0.0175 |               0.0328 |                0.2443 |             0.6348 |
| extra_trees  |     0.0156 |                    0.0283 |                  0.0192 |               0.0192 |               0.0308 |                0.2281 |             0.6512 |
| bayes_ridge  |     0.0157 |                    0.0389 |                  0.0163 |               0.0137 |               0.0410 |                0.3173 |             0.1312 |
| rf           |     0.0158 |                    0.0254 |                  0.0192 |               0.0213 |               0.0272 |                0.2036 |             0.4469 |
| enet         |     0.0158 |                    0.0364 |                  0.0165 |               0.0145 |               0.0316 |                0.2403 |             0.3371 |
| xgboost      |     0.0159 |                    0.0282 |                  0.0198 |               0.0204 |               0.0275 |                0.2108 |             0.3562 |
| pls          |     0.0161 |                    0.0360 |                  0.0167 |               0.0165 |               0.0360 |                0.2730 |             0.3156 |
| trend        |     0.0162 |                    0.0304 |                  0.0173 |               0.0158 |               0.0319 |                0.2370 |             0.7215 |
| mlp          |     0.0163 |                    0.0351 |                  0.0172 |               0.0153 |               0.0391 |                0.3002 |             0.5190 |
| midas        |     0.0163 |                    0.0306 |                  0.0174 |               0.0166 |               0.0234 |                0.1707 |             0.6268 |
| kernel_ridge |     0.0163 |                    0.0306 |                  0.0174 |               0.0165 |               0.0319 |                0.2360 |             0.7132 |
| hgb          |     0.0167 |                    0.0260 |                  0.0182 |               0.0203 |               0.0309 |                0.2206 |             0.4502 |
| gp           |     0.0167 |                    0.0302 |                  0.0179 |               0.0159 |               0.0313 |                0.2356 |             0.7132 |
| ard          |     0.0169 |                    0.0397 |                  0.0181 |               0.0156 |               0.0380 |                0.2932 |             0.1135 |

**Conclusion D2:** the inner CV does not measure what the forecast does, which explains both the flat losses and the MLP choice.

1. **Hindsight labels.** The labels are smoothed over all claims public at v, so early folds train on labels that did not exist at the block start. Own-pool folds with zero real-time training pairs: 2026-02-11 2 of 2, 2026-03-02 2 of 3, 2026-04-05 2 of 3, 2026-05-27 1 of 3, 2026-11-01 1 of 3, 2026-11-09 1 of 3. Anthropic had at most one anchor at those block starts (the Jan-2025 $1B claim was published only on 2025-09-02). Revisions of the training labels themselves are small (0.001-0.006/week), so the leak is mostly about which periods have labels at all.
2. **Interior validation pairs.** Validation pairs are interior windows of a smoothed path, where growth follows from the neighbouring slope, so every member with `mom` does about equally well. In the pooled pool, 6-7 of 18 members are within 10% of the best loss at the H2 vintages. At the median member, the real-task loss is 1.2x (own) and 2.0x (pooled) the inner-CV loss at the H2 vintages. At the spring vintages, which have only 3-4 real-task points, it is 0.45-2.37x.
3. **The rankings do not match.** Spearman correlation between inner-CV and real-task losses across members: 2026-11-09 pooled +0.06, 2026-11-09 own +0.17, 2026-11-01 pooled -0.07, 2026-11-01 own +0.15, 2026-05-27 pooled -0.64, 2026-05-27 own -0.22, 2026-04-05 pooled -0.56, 2026-04-05 own -0.19, 2026-03-02 pooled +0.45, 2026-03-02 own -0.64.
4. **Pooled CV is half OpenAI.** About half or more of the pooled validation rows are OpenAI, and the Anthropic-row loss of AUTO's pick is 1.1-2.1x the overall loss (April-November vintages, where OpenAI rows are 52%-54%). The members' ranks on Anthropic rows differ: at 2026-11-09 ridge, AUTO's pick, ranks 6 of 18 on Anthropic rows.
5. **The MLP.** In the own pool its inner-CV loss is the lowest at the late vintages, and clearly so (0.0207 vs 0.0234 next at 2026-11-09). With fold-nested labels it ranks 11th, and its loss rises most of all members (0.0227 to 0.0345 on the same folds). On the real task it ranks 14th (own) and 17th (pooled). Not tested: its early stopping uses a random 20% split of overlapping 4/8/13-week pairs, which would add the same kind of leak inside the fit.
6. **Nested CV alone does not give a selector.** With 3-14 real-task points per vintage, the nested-real-task pick (often midas or bayes_ridge) was no better on the test targets than AUTO's pick. Mean |err| over the 6 post-t0 targets of the 5 vintages with real-task points: pooled AUTO 0.213 vs nested 0.445; own AUTO 0.499 vs nested 0.500. With this little data, per-vintage selection among 18 members is the problem. Nested evaluation is still needed to compare designs honestly.

## D3. Late-starting features

Late-starting = missing in more than 50% of the Anthropic training pairs that start before 2026. Missing values become 0, which is the company mean after demeaning. Feature status at the 2026-11-09 vintage (pooled pairs):

| feature                           | how    |   miss_anth_early |   miss_anth_all |   miss_openai | first_anth_pair_t   |   interior_nan_weeks_zero_filled |
|:----------------------------------|:-------|------------------:|----------------:|--------------:|:--------------------|---------------------------------:|
| cloud_bedrock_growth_cum          | cumsum |              1.00 |            0.77 |          1.00 | 2026-02-05          |                            12.00 |
| cloud_foundry_growth_cum          | cumsum |              1.00 |            1.00 |          1.00 | nan                 |                             0.00 |
| consumer_app_downloads_dlog_cum   | cumsum |              1.00 |            1.00 |          1.00 | nan                 |                           nan    |
| consumer_app_downloads_share      | level  |              1.00 |            1.00 |          1.00 | nan                 |                           nan    |
| consumer_app_revenue_dlog_cum     | cumsum |              1.00 |            1.00 |          1.00 | nan                 |                           nan    |
| consumer_app_revenue_share        | level  |              1.00 |            1.00 |          1.00 | nan                 |                           nan    |
| consumer_web_visits_dlog_cum      | cumsum |              1.00 |            1.00 |          1.00 | nan                 |                             0.00 |
| consumer_web_visits_share         | level  |              1.00 |            0.93 |          0.82 | 2026-04-16          |                           nan    |
| github_opus_share                 | level  |              0.93 |            0.64 |          1.00 | 2025-12-11          |                           nan    |
| langchain_pypi_share              | level  |              1.00 |            0.91 |          0.81 | 2026-04-09          |                           nan    |
| npm_agent_sdk_dlog_cum            | cumsum |              0.73 |            0.50 |          1.00 | 2025-10-09          |                             3.00 |
| openrouter_google_dlog_tokens_cum | cumsum |              0.89 |            0.61 |          0.58 | 2025-11-27          |                             0.00 |
| openrouter_google_share           | level  |              0.86 |            0.59 |          0.56 | 2025-11-20          |                           nan    |
| openrouter_log_tokens             | level  |              0.86 |            0.59 |          0.56 | 2025-11-20          |                           nan    |
| openrouter_share                  | level  |              0.86 |            0.59 |          0.56 | 2025-11-20          |                           nan    |
| pypi_level                        | cumsum |              1.00 |            0.93 |          0.82 | 2026-04-16          |                             0.00 |
| vercel_spend_share                | level  |              1.00 |            0.97 |          0.94 | 2026-05-14          |                           nan    |

Coefficients and contributions of the late features in AUTO's chosen member at the H2-2026 vintages (feature set `all`; pooled = lasso, own = MLP):

- `contrib_rate` is the change in predicted weekly growth when the feature at the forecast row is set to missing (= company mean).
- `z_forecast_absmax` is the largest |standardized value| at the forecast row; `z_train_p99` is the 99th percentile in the present training rows.
- `perm_importance` is the increase in training loss when the column is permuted.

|                                                          |   coef_std |   contrib_rate_mean |   contrib_rate_absmax |   z_forecast_absmax |   z_train_p99 |   perm_importance |
|:---------------------------------------------------------|-----------:|--------------------:|----------------------:|--------------------:|--------------:|------------------:|
| ('own', 'mlp', 'cloud_bedrock_growth_cum')               |   nan      |              0.0020 |                0.0105 |              7.8527 |        4.5795 |            0.0028 |
| ('own', 'mlp', 'consumer_web_visits_share')              |   nan      |              0.0130 |                0.0201 |             11.9963 |       13.8535 |            0.0012 |
| ('own', 'mlp', 'github_opus_share')                      |   nan      |             -0.0000 |                0.0018 |              2.1334 |        4.3792 |            0.0011 |
| ('own', 'mlp', 'langchain_pypi_share')                   |   nan      |              0.0130 |                0.0213 |              9.3019 |       16.1842 |            0.0009 |
| ('own', 'mlp', 'npm_agent_sdk_dlog_cum')                 |   nan      |              0.0062 |                0.0139 |              2.1654 |        4.3509 |            0.0015 |
| ('own', 'mlp', 'openrouter_google_dlog_tokens_cum')      |   nan      |             -0.0002 |                0.0045 |              3.0223 |        5.0210 |            0.0007 |
| ('own', 'mlp', 'openrouter_google_share')                |   nan      |              0.0022 |                0.0044 |              6.4258 |        6.6326 |            0.0025 |
| ('own', 'mlp', 'openrouter_log_tokens')                  |   nan      |             -0.0195 |                0.0407 |             10.3058 |        6.0880 |            0.0021 |
| ('own', 'mlp', 'openrouter_share')                       |   nan      |              0.0138 |                0.0310 |             22.8641 |        6.6658 |            0.0007 |
| ('own', 'mlp', 'pypi_level')                             |   nan      |             -0.0001 |                0.0022 |              3.7448 |       20.0357 |            0.0005 |
| ('own', 'mlp', 'vercel_spend_share')                     |   nan      |              0.0038 |                0.0166 |             87.6750 |       29.4292 |            0.0002 |
| ('pooled', 'lasso', 'cloud_bedrock_growth_cum')          |     0.0000 |              0.0000 |                0.0000 |             14.3049 |        8.3466 |            0.0000 |
| ('pooled', 'lasso', 'consumer_web_visits_share')         |     0.0000 |              0.0000 |                0.0000 |              7.1351 |       14.0009 |            0.0000 |
| ('pooled', 'lasso', 'github_opus_share')                 |     0.0000 |              0.0000 |                0.0000 |              3.8882 |        7.9635 |            0.0000 |
| ('pooled', 'lasso', 'langchain_pypi_share')              |     0.0000 |              0.0000 |                0.0000 |              7.2112 |       14.2522 |            0.0000 |
| ('pooled', 'lasso', 'npm_agent_sdk_dlog_cum')            |     0.0000 |              0.0000 |                0.0000 |              3.9224 |        7.9223 |            0.0000 |
| ('pooled', 'lasso', 'openrouter_google_dlog_tokens_cum') |     0.0000 |              0.0000 |                0.0000 |              3.0698 |        5.0606 |            0.0000 |
| ('pooled', 'lasso', 'openrouter_google_share')           |     0.0000 |              0.0000 |                0.0000 |              7.2266 |        8.8076 |            0.0000 |
| ('pooled', 'lasso', 'openrouter_log_tokens')             |     0.0000 |              0.0000 |                0.0000 |              9.6473 |        7.5596 |            0.0000 |
| ('pooled', 'lasso', 'openrouter_share')                  |     0.0000 |              0.0000 |                0.0000 |             22.5269 |        7.4937 |            0.0000 |
| ('pooled', 'lasso', 'pypi_level')                        |     0.0000 |              0.0000 |                0.0000 |              1.9186 |       16.1341 |            0.0000 |
| ('pooled', 'lasso', 'vercel_spend_share')                |     0.0000 |              0.0000 |                0.0000 |            159.7949 |       49.3219 |            0.0000 |

Refits of the same member and hyperparameters at the H2 targets:

- `err_no_late`: the late features dropped.
- `err_gspec_cols`: only the g_spec columns.
- `err_complete_case`: training only on pairs starting after every late feature exists, which leaves 16-38 pooled and 6 own pairs, so it is too small to use.
- `y_mean_late_*`: Anthropic label growth/week in pairs where all late features are missing vs present.

|                                                       | member      |   err_full |   err_no_late |   err_gspec_cols |   err_complete_case |   n_pairs_complete_case |   y_mean_late_missing |   y_mean_late_present |
|:------------------------------------------------------|:------------|-----------:|--------------:|-----------------:|--------------------:|------------------------:|----------------------:|----------------------:|
| ('all', 'own', 'period_end', 'rev_2026q3')            | mlp         |     -0.351 |        -0.214 |           -0.237 |              -0.260 |                       6 |                 0.039 |                 0.067 |
| ('all', 'own', 'period_end', 'rr_2026_09_s1a')        | mlp         |     -0.648 |        -0.392 |           -0.459 |              -0.529 |                       6 |                 0.039 |                 0.067 |
| ('all', 'own', 'period_end', 'rr_2026_10_s1a')        | mlp         |     -1.041 |        -0.169 |           -0.786 |              -0.609 |                       6 |                 0.039 |                 0.067 |
| ('all', 'own', 'publication', 'rev_2026q3')           | mlp         |     -0.392 |        -0.213 |           -0.241 |              -0.260 |                       6 |                 0.039 |                 0.067 |
| ('all', 'own', 'publication', 'rr_2026_09_s1a')       | mlp         |     -0.576 |        -0.426 |           -0.488 |              -0.581 |                       6 |                 0.039 |                 0.068 |
| ('all', 'own', 'publication', 'rr_2026_10_s1a')       | mlp         |     -0.633 |        -0.208 |           -0.710 |              -0.674 |                       6 |                 0.039 |                 0.068 |
| ('all', 'pooled', 'period_end', 'rev_2026q3')         | lasso       |     -0.214 |        -0.214 |           -0.218 |              -0.224 |                      16 |                 0.039 |                 0.067 |
| ('all', 'pooled', 'period_end', 'rr_2026_09_s1a')     | lasso       |     -0.450 |        -0.450 |           -0.457 |              -0.469 |                      16 |                 0.039 |                 0.067 |
| ('all', 'pooled', 'period_end', 'rr_2026_10_s1a')     | lasso       |     -0.351 |        -0.351 |           -0.366 |              -0.655 |                      38 |                 0.039 |                 0.067 |
| ('all', 'pooled', 'publication', 'rev_2026q3')        | lasso       |     -0.150 |        -0.150 |           -0.143 |              -0.281 |                      38 |                 0.039 |                 0.067 |
| ('all', 'pooled', 'publication', 'rr_2026_09_s1a')    | lasso       |     -0.370 |        -0.370 |           -0.363 |              -0.603 |                      38 |                 0.039 |                 0.068 |
| ('all', 'pooled', 'publication', 'rr_2026_10_s1a')    | lasso       |     -0.388 |        -0.388 |           -0.399 |              -0.727 |                      38 |                 0.039 |                 0.068 |
| ('g_spec', 'own', 'period_end', 'rev_2026q3')         | mlp         |     -0.411 |        -0.205 |           -0.411 |              -0.255 |                       6 |                 0.039 |                 0.074 |
| ('g_spec', 'own', 'period_end', 'rr_2026_09_s1a')     | mlp         |     -0.703 |        -0.399 |           -0.703 |              -0.509 |                       6 |                 0.039 |                 0.074 |
| ('g_spec', 'own', 'period_end', 'rr_2026_10_s1a')     | mlp         |     -1.121 |        -0.454 |           -1.121 |              -0.571 |                       6 |                 0.039 |                 0.074 |
| ('g_spec', 'own', 'publication', 'rev_2026q3')        | mlp         |     -0.408 |        -0.206 |           -0.408 |              -0.255 |                       6 |                 0.039 |                 0.074 |
| ('g_spec', 'own', 'publication', 'rr_2026_09_s1a')    | mlp         |     -0.729 |        -0.439 |           -0.729 |              -0.570 |                       6 |                 0.039 |                 0.074 |
| ('g_spec', 'own', 'publication', 'rr_2026_10_s1a')    | mlp         |     -1.145 |        -0.499 |           -1.145 |              -0.652 |                       6 |                 0.039 |                 0.074 |
| ('g_spec', 'pooled', 'period_end', 'rev_2026q3')      | extra_trees |     -0.267 |        -0.270 |           -0.267 |              -0.160 |                      16 |                 0.039 |                 0.074 |
| ('g_spec', 'pooled', 'period_end', 'rr_2026_09_s1a')  | extra_trees |     -0.526 |        -0.531 |           -0.526 |              -0.353 |                      16 |                 0.039 |                 0.074 |
| ('g_spec', 'pooled', 'period_end', 'rr_2026_10_s1a')  | ridge       |     -0.184 |        -0.254 |           -0.184 |              -0.509 |                      38 |                 0.039 |                 0.074 |
| ('g_spec', 'pooled', 'publication', 'rev_2026q3')     | ridge       |     -0.041 |        -0.099 |           -0.041 |              -0.224 |                      38 |                 0.039 |                 0.074 |
| ('g_spec', 'pooled', 'publication', 'rr_2026_09_s1a') | ridge       |     -0.233 |        -0.314 |           -0.233 |              -0.527 |                      38 |                 0.039 |                 0.074 |
| ('g_spec', 'pooled', 'publication', 'rr_2026_10_s1a') | ridge       |     -0.256 |        -0.295 |           -0.256 |              -0.598 |                      38 |                 0.039 |                 0.074 |

Item 3, AUTO with 43 vs 12 columns (publication): the change in |err| split into the feature effect (the member AUTO picked with g_spec, refit with all features) and the effect of AUTO switching members:

|                              |   gap_weeks | AUTO g_spec picks   | AUTO all picks   |   err g_spec |   err all |   err all, g_spec's member |   d|err| total |   of which: features (same member) |   of which: AUTO switched member |
|:-----------------------------|------------:|:--------------------|:-----------------|-------------:|----------:|---------------------------:|---------------:|-----------------------------------:|---------------------------------:|
| ('pooled', 'rev_2026q3')     |      11.140 | ridge               | lasso            |       -0.041 |    -0.150 |                      0.066 |          0.109 |                              0.025 |                            0.084 |
| ('pooled', 'rr_2026_02')     |      15.430 | pca                 | pca              |        0.359 |     0.325 |                      0.325 |         -0.034 |                             -0.034 |                            0.000 |
| ('pooled', 'rr_2026_03')     |       1.860 | rf                  | rf               |        0.238 |     0.238 |                      0.238 |          0.000 |                              0.000 |                            0.000 |
| ('pooled', 'rr_2026_04')     |       3.710 | rf                  | rf               |        0.379 |     0.387 |                      0.387 |          0.008 |                              0.008 |                            0.000 |
| ('pooled', 'rr_2026_05')     |       8.000 | kernel_ridge        | midas            |        0.128 |    -0.537 |                      0.148 |          0.408 |                              0.019 |                            0.389 |
| ('pooled', 'rr_2026_09_s1a') |       9.140 | ridge               | lasso            |       -0.233 |    -0.370 |                      0.001 |          0.136 |                             -0.232 |                            0.368 |
| ('pooled', 'rr_2026_10_s1a') |      13.570 | ridge               | lasso            |       -0.256 |    -0.388 |                      0.122 |          0.132 |                             -0.133 |                            0.265 |
| ('own', 'rev_2026q3')        |      11.140 | mlp                 | mlp              |       -0.408 |    -0.392 |                     -0.392 |         -0.015 |                             -0.015 |                            0.000 |
| ('own', 'rr_2026_02')        |      15.430 | trend               | trend            |        0.360 |     0.360 |                      0.360 |          0.000 |                              0.000 |                            0.000 |
| ('own', 'rr_2026_03')        |       1.860 | mlp                 | pca              |        0.214 |     0.160 |                      0.215 |         -0.054 |                              0.001 |                           -0.055 |
| ('own', 'rr_2026_04')        |       3.710 | pca                 | pls              |        0.458 |     0.596 |                      0.496 |          0.137 |                              0.038 |                            0.099 |
| ('own', 'rr_2026_05')        |       8.000 | mlp                 | midas            |       -0.039 |    -0.467 |                     -0.187 |          0.427 |                              0.148 |                            0.280 |
| ('own', 'rr_2026_09_s1a')    |       9.140 | mlp                 | mlp              |       -0.729 |    -0.576 |                     -0.576 |         -0.153 |                             -0.153 |                            0.000 |
| ('own', 'rr_2026_10_s1a')    |      13.570 | mlp                 | mlp              |       -1.145 |    -0.633 |                     -0.633 |         -0.512 |                             -0.512 |                            0.000 |

Mean over the 14 publication targets (the 7 pre-t0 targets contribute 0):

| pool   |   d|err| total |   of which: features (same member) |   of which: AUTO switched member |
|:-------|---------------:|-----------------------------------:|---------------------------------:|
| own    |         -0.012 |                             -0.035 |                            0.023 |
| pooled |          0.054 |                             -0.025 |                            0.079 |

**Conclusion D3:**

- **Lasso: no effect.** The pooled lasso, AUTO's pick with `all`, sets every late-starting feature to exactly 0, and dropping them changes nothing (err_full = err_no_late at all 6 H2 cases). It mainly uses github_repo_share, ramp_vendor_share, langchain_npm_share and ramp_adoption.
- **MLP: a spurious pattern.** Imputation creates one for the MLP. The late features are present only from Nov 2025 to Jul 2026, the hyper-growth era (0.067/week vs 0.039 where they are missing). Scaled by the RMS over mostly-imputed rows, they reach the forecast row far outside the training range (vercel_spend_share z = -88 vs p99 29; openrouter_share 22.9 vs 6.7; cloud_bedrock_growth_cum 7.9 vs 4.6). The MLP turns them into net extra growth of about +0.03/week (sum of contrib_rate).
- **The size of the MLP effect.** Dropping the late features improves the own-pool MLP at all 12 H2 cases. For example, rr_2026_10 with `all` goes from -1.041 to -0.169 (period-end). With g_spec, the +214% case (-1.145) becomes -0.499: OpenRouter tokens/share from Nov 2025, vercel from May 2026 and pypi from Apr 2026 are late-starting even in g_spec.
- **Linear models are fine.** For the pooled ridge the same g_spec late features carry real signal (dropping them worsens rr_2026_09 from -0.233 to -0.314).
- **Item 3 is a selection effect.** Going from 12 to 43 columns improves the same member slightly. AUTO's loss comes from switching to a near-tied member: midas at rr_2026_05 (-0.537 vs kernel_ridge +0.128) and lasso instead of ridge at the three H2 targets.

## D4. Pre-t0 targets: why the label path is worse than G's path

Each step changes one thing, using `backtest.prepublication_forecast` exactly as the A/G scripts do (refit=True, catalog roles patched as in `forecast_ramp.py gh_share`):

- **S0:** the zoo's value: the label path, exp(mean) integrated. It is the same for every member.
- **S1:** the panel smoother's own forecast (the mean).
- **S1r:** the panel replica inside the backtest machinery (panel noise, rho 0.5), as a check.
- **S2:** the backtest's process-noise refit.
- **S3:** plus the crossing profile (= A).
- **S4:** plus segment claims (Claude Code run-rates, shares).
- **S5:** plus the GitHub new-repo share indicator (= G).
- **S5b:** G without segment claims.

The definition lag is the same in both pipelines (`config.DEFINITION_LAG_DAYS`, t28x13 = 14 days). The target dates (`t_eff`) of these 7 targets are identical under every rho (none is a crossing claim).

|                  |   S0 zoo label path (every member) |   S1 SS panel (smoother mean) |   S1r panel replica (panel noise, rho .5) |   S2 + backtest noise refit |   S3 + crossing profile (= A) |   S4 + segment claims |   S5 + GitHub share indicator (= G) |   S5b G without segment claims |   G (CSV) |   A (CSV) |   E (CSV) |
|:-----------------|-----------------------------------:|------------------------------:|------------------------------------------:|----------------------------:|------------------------------:|----------------------:|------------------------------------:|-------------------------------:|----------:|----------:|----------:|
| rr_2025_12       |                             -0.171 |                        -0.171 |                                    -0.171 |                      -0.171 |                        -0.163 |                -0.145 |                              -0.039 |                         -0.130 |    -0.039 |    -0.163 |    -0.003 |
| rev_2025q1       |                              0.098 |                         0.091 |                                     0.091 |                       0.091 |                         0.082 |                 0.083 |                              -0.023 |                         -0.024 |    -0.023 |     0.082 |     0.166 |
| rev_2025q2       |                              0.028 |                         0.020 |                                     0.020 |                       0.020 |                         0.067 |                 0.053 |                              -0.012 |                         -0.006 |    -0.012 |     0.067 |     0.029 |
| rev_2025q3       |                              0.041 |                         0.030 |                                     0.030 |                       0.030 |                         0.043 |                 0.021 |                              -0.042 |                         -0.027 |    -0.042 |     0.043 |    -0.184 |
| rev_2025fy       |                              0.080 |                         0.073 |                                     0.073 |                       0.073 |                         0.062 |                 0.060 |                              -0.022 |                         -0.024 |    -0.022 |     0.062 |     0.014 |
| rev_2026q1       |                              0.044 |                         0.044 |                                     0.044 |                       0.044 |                         0.050 |                 0.046 |                              -0.050 |                         -0.045 |    -0.050 |     0.050 |     0.048 |
| rev_2026q2       |                             -0.010 |                        -0.010 |                                    -0.010 |                      -0.010 |                        -0.004 |                -0.004 |                               0.005 |                          0.005 |     0.005 |    -0.004 |    -0.008 |
| mean |log error| |                              0.067 |                         0.063 |                                     0.063 |                       0.063 |                         0.067 |                 0.059 |                               0.028 |                          0.037 |     0.028 |     0.067 |     0.064 |

Change in |err| at each step (+ = worse):

|                          |   S0->S1 |   S1->S1r |   S1r->S2 |   S2->S3 |   S3->S4 |   S4->S5 |   S0->S5 total |
|:-------------------------|---------:|----------:|----------:|---------:|---------:|---------:|---------------:|
| rr_2025_12               |    0.000 |     0.000 |    -0.000 |   -0.009 |   -0.018 |   -0.106 |         -0.132 |
| rev_2025q1               |   -0.007 |     0.000 |     0.000 |   -0.008 |    0.001 |   -0.060 |         -0.075 |
| rev_2025q2               |   -0.008 |     0.000 |     0.000 |    0.047 |   -0.014 |   -0.041 |         -0.016 |
| rev_2025q3               |   -0.011 |     0.000 |     0.000 |    0.013 |   -0.022 |    0.022 |          0.002 |
| rev_2025fy               |   -0.007 |     0.000 |     0.000 |   -0.010 |   -0.002 |   -0.039 |         -0.058 |
| rev_2026q1               |    0.000 |     0.000 |     0.000 |    0.006 |   -0.003 |    0.003 |          0.006 |
| rev_2026q2               |    0.000 |     0.000 |    -0.000 |   -0.006 |    0.000 |    0.001 |         -0.005 |
| mean (signed: + = worse) |   -0.005 |     0.000 |     0.000 |    0.005 |   -0.008 |   -0.031 |         -0.040 |

Process noise at the two vintages (panel `fit_noise` vs `calibrate.fit_process_noise`):

|                              |   2026-10-25 |   2026-03-02 |
|:-----------------------------|-------------:|-------------:|
| noise_panel:sigma_l          |       0.0042 |       0.0043 |
| noise_ss:sigma_l             |       0.0042 |       0.0043 |
| noise_panel:sigma_g          |       0.0218 |       0.0067 |
| noise_ss:sigma_g             |       0.0218 |       0.0067 |
| noise_panel:sigma_b          |       0.0020 |       0.0021 |
| noise_ss:sigma_b             |       0.0020 |       0.0021 |
| noise_panel:growth_reversion |       0.1385 |       0.0815 |
| noise_ss:growth_reversion    |       0.1385 |       0.0815 |
| rho_A                        |       0.9197 |       0.7286 |
| rho_G                        |       0.8879 |       0.5984 |

**Conclusion D4:**

- **The steps that contribute nothing or little.** The two process-noise fits agree to 4 digits (contribution 0). The crossing profile moves individual quarters by up to 0.05 in either direction (rho 0.89-0.92 vs 0.5) and averages +0.005. Segment claims contribute -0.008. The Jensen term costs the zoo path 0 to 0.011 on the revenue targets (S0 vs S1).
- **The indicator is the whole difference.** Adding the GitHub share indicator moves the mean |err| from 0.059 to 0.028. The zoo path's mean is 0.067.
- **Why the indicator helps.** Before t0 the claims-only path is a near log-linear interpolation between 3-10 anchors: Jan 2025 ($1B), May 2025, Oct 2025, Dec 2025, Jan/Feb/Mar 2026 and so on. The indicator bends it between anchors. It raises 2025 revenue (all G errors turn slightly negative) and lowers December 2025 (rr_2025_12: -0.145 to -0.039). Segment claims and the indicator interact: G without segments gives -0.130 on rr_2025_12.
- **Caveat on the GitHub series.** It was fetched in one go on 2026-11-11 and includes repos made public later, so its pre-t0 advantage partly reflects hindsight data.
- **No member can do better.** No ML member can improve these targets in the current design: they only measure the label smoother.

