# Corrected ML pipeline (ml_fixed.py): backtest (2026-11-10 data)

Specification: the docstring of `runrate_nowcast/ml_fixed.py`, fixed before the first run. Two pre-declared
models: **M1** = ridge toward prior elasticities (1 on OpenRouter tokens x price, log GitHub new-repo
share and log Ramp adoption; 0 on the other g_spec features and on momentum), penalty chosen by
nested-vintage CV (fold labels rebuilt from claims published before each validation block, 13-week
purge); **M2** = equal-weight log average of M1 and the claims-only state-space path. Features
standardized on observed rows, clipped to the training 1st-99th percentile, dropped with < 26 observed
weeks. Crossing claims re-dated at the rho profiled at each vintage. `pooled` = Anthropic + OpenAI pairs
(primary); `own` = target company only (control). Errors are log(actual / forecast); negative = forecast
too high. Filing rows are never inputs.

## Reading the results

- **Post-t0 Anthropic targets, mean |log error|:** publication: M1 0.420, M2 0.168, E 0.145, G 0.171, A 0.347, old AUTO 0.234 (n = 7); period_end: M1 0.286, M2 0.122, E 0.156, G 0.151, A 0.224, old AUTO 0.224 (n = 11).
- **Why M1 fails.** Wherever the nested CV picks the strongest prior (lambda 1e6), M1 is the prior itself:
  company mean growth + 1 x (each proxy's weekly change - its training mean), summed over the three proxies.
  At the H2-2026 targets the company mean is 0.060/week, and the three proxy terms sum to -0.156/week
  (OpenRouter tokens x price alone -0.101), so M1 predicts -0.096/week after t0 and is far too low.
  Two parts of the specification cause it. First, a prior of 1 on each of three proxies of the same
  revenue adds their deviations; the state space averages them as separate measurements of one level.
  Second, the company effect carries each proxy's historical excess growth over revenue (OpenRouter
  tokens grew faster than revenue in its training window), which the state space's drifting
  intercept does not. The ablation `M1 prior 1/3`, which averages the proxies instead of summing them,
  is the best ML variant here.
- **M2 (publication) gains by cancelling errors.** M1 and the claims-only path err in opposite directions on 4 of 7 post-t0 targets (bias M1 +0.420, SS -0.151; + = too low). M2's 0.168 comes from averaging two biased forecasts, not from an accurate M1.
- **M2 (period_end) gains by cancelling errors.** M1 and the claims-only path err in opposite directions on 6 of 11 post-t0 targets (bias M1 +0.271, SS -0.092; + = too low). M2's 0.122 comes from averaging two biased forecasts, not from an accurate M1.
- **The nested CV has little to work with.** Anthropic vintages have 0-2 usable folds: none up to mid-February 2026, then one or two (the early blocks fail the usability rule of spec item 4). From April 2026 on the CV picks lambda 1e6 (1e4 in May), so M1 is the prior.
- **Ablations (pooled, post-t0; differences are paired, negative = the ablation is better than M1):** see the
  table above. Prior 0, the old inner CV and the old preprocessing all beat M1 (the intervals exclude 0,
  except the inner CV under period_end), because each one weakens the wrong prior. The old
  preprocessing scales sparse features by their RMS over imputed zeros, which
  shrinks their scale. A given lambda then penalizes the elasticity less, and the CV picks lambda 1-100,
  which leaves elasticities of about 0.6 on or_spend. The crossing fix helps a little: `M1 rho 0.5` is worse
  by about 0.03 in both rules, with an interval that excludes 0. The pre-t0 targets are identical for
  every ML variant.
- **OpenAI (second test set):** M1 and M2 beat the claims-only path, the naive last claim and the old AUTO under
  the publication rule; under period_end the gains over the path and the old AUTO are small (intervals
  include 0 for M1). For OpenAI the nested CV has up to 3 folds and picks lambda 10-1000 at 14 of 15
  targets in each rule, so M1 there is mostly fitted from data, not the summed prior.

## Changes after the specification was written

- Before the first backtest (no score seen): (a) rho = 0.5 when a company has no claim public at a fold
  vintage (a unit test failed on it; no labels exist there); (b) the `M1 prior 1/3` ablation and the
  contribution columns, after the inputs of one forecast showed the summed prior (-0.13/week at
  rr_2026_09_s1a). M1 and M2 are as specified.
- First backtest launch: crashed in a worker before writing any output (no score produced). The pooled-pairs
  assertion compared each pool company's `t1` with its `t0`, which is None for Anthropic at early OpenAI
  vintages (it contributes no pairs then). Fixed to check the companies present in the pairs, as
  panel_model does. A regression test was added. The results here come from the second launch, the
  only run that produced results. No bug was fixed after results existed, so there are no before/after
  numbers.

## Anthropic targets

### Headline: mean |log error| on the common Anthropic targets

Common targets (per rule) = those forecast by M1, M2 (pooled), A, E, G, the naive last claim and the old
zoo AUTO (pooled). Post-t0 = the target lies after the last anchor public at the vintage (primary); `all`
adds the pre-t0 targets, where every ML model returns the claims-only label path.

| model                       |   publication: n post-t0 |   publication: mean |err| post-t0 |   publication: n all |   publication: mean |err| all |   period_end: n post-t0 |   period_end: mean |err| post-t0 |   period_end: n all |   period_end: mean |err| all |
|:----------------------------|-------------------------:|----------------------------------:|---------------------:|------------------------------:|------------------------:|---------------------------------:|--------------------:|-----------------------------:|
| M1 (pooled)                 |                        7 |                             0.42  |                   14 |                         0.245 |                      11 |                            0.286 |                  11 |                        0.286 |
| M2 (pooled)                 |                        7 |                             0.168 |                   14 |                         0.118 |                      11 |                            0.122 |                  11 |                        0.122 |
| M1 (own)                    |                        7 |                             0.328 |                   14 |                         0.199 |                      11 |                            0.232 |                  11 |                        0.232 |
| M2 (own)                    |                        7 |                             0.123 |                   14 |                         0.096 |                      11 |                            0.094 |                  11 |                        0.094 |
| SS claims-only path         |                        7 |                             0.347 |                   14 |                         0.207 |                      11 |                            0.224 |                  11 |                        0.224 |
| A claims only               |                        7 |                             0.347 |                   14 |                         0.207 |                      11 |                            0.224 |                  11 |                        0.224 |
| E OpenRouter tokens x price |                        7 |                             0.145 |                   14 |                         0.105 |                      11 |                            0.156 |                  11 |                        0.156 |
| G GitHub new-repo share     |                        7 |                             0.171 |                   14 |                         0.099 |                      11 |                            0.151 |                  11 |                        0.151 |
| naive last claim            |                        7 |                             0.296 |                   14 |                         0.221 |                      11 |                            0.257 |                  11 |                        0.257 |
| old AUTO (pooled)           |                        7 |                             0.234 |                   14 |                         0.15  |                      11 |                            0.224 |                  11 |                        0.224 |
| old AUTO (own)              |                        7 |                             0.479 |                   14 |                         0.273 |                      11 |                            0.337 |                  11 |                        0.337 |

### Vintage rule: publication

Scores with bias (mean log error on post-t0 targets; + = forecast too low):

| model                       |   n post-t0 |   mean |err| post-t0 |   bias post-t0 |   n all |   mean |err| all |
|:----------------------------|------------:|---------------------:|---------------:|--------:|-----------------:|
| M1 (pooled)                 |           7 |                0.42  |          0.42  |      14 |            0.245 |
| M2 (pooled)                 |           7 |                0.168 |          0.134 |      14 |            0.118 |
| M1 (own)                    |           7 |                0.328 |          0.328 |      14 |            0.199 |
| M2 (own)                    |           7 |                0.123 |          0.089 |      14 |            0.096 |
| SS claims-only path         |           7 |                0.347 |         -0.151 |      14 |            0.207 |
| A claims only               |           7 |                0.347 |         -0.151 |      14 |            0.207 |
| E OpenRouter tokens x price |           7 |                0.145 |          0.062 |      14 |            0.105 |
| G GitHub new-repo share     |           7 |                0.171 |          0.001 |      14 |            0.099 |
| naive last claim            |           7 |                0.296 |          0.285 |      14 |            0.221 |
| old AUTO (pooled)           |           7 |                0.234 |          0.082 |      14 |            0.15  |
| old AUTO (own)              |           7 |                0.479 |         -0.184 |      14 |            0.273 |

Paired differences in |log error| (`B vs A`: negative = B better; 90% bootstrap interval;
forecast_eval.paired):

| targets   | rule        | comparison                                 |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:----------|:------------|:-------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| post-t0   | publication | M1 (pooled) vs E OpenRouter tokens x price |   7 |       0.275 |     0.127 |     0.458 |          1 |       6 |
| post-t0   | publication | M1 (pooled) vs G GitHub new-repo share     |   7 |       0.249 |     0.094 |     0.436 |          1 |       6 |
| post-t0   | publication | M1 (pooled) vs A claims only               |   7 |       0.073 |    -0.051 |     0.217 |          3 |       4 |
| post-t0   | publication | M1 (pooled) vs old AUTO (pooled)           |   7 |       0.186 |    -0.013 |     0.411 |          3 |       4 |
| post-t0   | publication | M1 (pooled) vs naive last claim            |   7 |       0.123 |    -0.182 |     0.461 |          4 |       3 |
| post-t0   | publication | M2 (pooled) vs E OpenRouter tokens x price |   7 |       0.024 |    -0.061 |     0.106 |          2 |       5 |
| post-t0   | publication | M2 (pooled) vs G GitHub new-repo share     |   7 |      -0.003 |    -0.08  |     0.072 |          4 |       3 |
| post-t0   | publication | M2 (pooled) vs A claims only               |   7 |      -0.179 |    -0.317 |    -0.053 |          5 |       2 |
| post-t0   | publication | M2 (pooled) vs old AUTO (pooled)           |   7 |      -0.065 |    -0.125 |    -0.009 |          4 |       3 |
| post-t0   | publication | M2 (pooled) vs naive last claim            |   7 |      -0.128 |    -0.263 |     0.012 |          5 |       2 |
| post-t0   | publication | M1 (pooled) vs M1 (own)                    |   7 |       0.091 |     0.03  |     0.165 |          1 |       6 |
| post-t0   | publication | M2 (pooled) vs M2 (own)                    |   7 |       0.045 |     0.014 |     0.082 |          1 |       6 |
| post-t0   | publication | M2 (pooled) vs M1 (pooled)                 |   7 |      -0.251 |    -0.453 |    -0.066 |          5 |       2 |
| post-t0   | publication | M2 (pooled) vs SS claims-only path         |   7 |      -0.179 |    -0.317 |    -0.054 |          5 |       2 |
| all       | publication | M1 (pooled) vs E OpenRouter tokens x price |  14 |       0.14  |     0.046 |     0.251 |          4 |      10 |
| all       | publication | M1 (pooled) vs G GitHub new-repo share     |  14 |       0.146 |     0.06  |     0.249 |          3 |      11 |
| all       | publication | M1 (pooled) vs A claims only               |  14 |       0.038 |    -0.024 |     0.115 |          4 |      10 |
| all       | publication | M1 (pooled) vs old AUTO (pooled)           |  14 |       0.094 |    -0.008 |     0.217 |          7 |       7 |
| all       | publication | M1 (pooled) vs naive last claim            |  14 |       0.024 |    -0.132 |     0.202 |         11 |       3 |
| all       | publication | M2 (pooled) vs E OpenRouter tokens x price |  14 |       0.014 |    -0.037 |     0.065 |          5 |       9 |
| all       | publication | M2 (pooled) vs G GitHub new-repo share     |  14 |       0.019 |    -0.025 |     0.06  |          6 |       8 |
| all       | publication | M2 (pooled) vs A claims only               |  14 |      -0.089 |    -0.172 |    -0.018 |          6 |       8 |
| all       | publication | M2 (pooled) vs old AUTO (pooled)           |  14 |      -0.032 |    -0.067 |    -0.001 |          8 |       6 |
| all       | publication | M2 (pooled) vs naive last claim            |  14 |      -0.102 |    -0.173 |    -0.032 |         12 |       2 |
| all       | publication | M1 (pooled) vs M1 (own)                    |  14 |       0.046 |     0.011 |     0.088 |          1 |       6 |
| all       | publication | M2 (pooled) vs M2 (own)                    |  14 |       0.023 |     0.005 |     0.044 |          1 |       6 |
| all       | publication | M2 (pooled) vs M1 (pooled)                 |  14 |      -0.126 |    -0.246 |    -0.028 |         10 |       3 |
| all       | publication | M2 (pooled) vs SS claims-only path         |  14 |      -0.089 |    -0.169 |    -0.019 |          6 |       7 |

Per-target log error (+ = forecast too low):

| target         | post_t0   |   M1 (pooled) |   M2 (pooled) |   M1 (own) |   M2 (own) |   SS claims-only path |   A claims only |   E OpenRouter tokens x price |   G GitHub new-repo share |   naive last claim |   old AUTO (pooled) |   old AUTO (own) |
|:---------------|:----------|--------------:|--------------:|-----------:|-----------:|----------------------:|----------------:|------------------------------:|--------------------------:|-------------------:|--------------------:|-----------------:|
| rev_2026q3     | True      |         0.45  |         0.063 |      0.342 |      0.009 |                -0.324 |          -0.324 |                         0.051 |                     0.014 |             -0.038 |              -0.041 |           -0.408 |
| rr_2026_02     | True      |         0.249 |         0.306 |      0.103 |      0.233 |                 0.363 |           0.363 |                         0.096 |                     0.113 |              0.693 |               0.359 |            0.36  |
| rr_2026_03     | True      |         0.271 |         0.244 |      0.213 |      0.215 |                 0.217 |           0.217 |                         0.087 |                     0.18  |              0.305 |               0.238 |            0.214 |
| rr_2026_04     | True      |         0.246 |         0.177 |      0.246 |      0.177 |                 0.108 |           0.108 |                         0.108 |                     0.197 |              0.457 |               0.379 |            0.458 |
| rr_2026_05     | True      |         0.06  |        -0.073 |      0.057 |     -0.075 |                -0.206 |          -0.206 |                         0.12  |                     0.098 |              0.449 |               0.128 |           -0.039 |
| rr_2026_09_s1a | True      |         1.11  |         0.271 |      0.785 |      0.108 |                -0.568 |          -0.568 |                         0.262 |                    -0.291 |              0     |              -0.233 |           -0.729 |
| rr_2026_10_s1a | True      |         0.551 |        -0.046 |      0.553 |     -0.045 |                -0.642 |          -0.642 |                        -0.288 |                    -0.302 |              0.132 |              -0.256 |           -1.145 |
| rev_2025fy     | False     |         0.067 |         0.065 |      0.067 |      0.065 |                 0.062 |           0.062 |                         0.014 |                    -0.022 |              0.118 |               0.08  |            0.08  |
| rev_2025q1     | False     |         0.088 |         0.085 |      0.088 |      0.085 |                 0.082 |           0.082 |                         0.166 |                    -0.023 |              0.145 |               0.098 |            0.098 |
| rev_2025q2     | False     |         0.072 |         0.069 |      0.072 |      0.069 |                 0.067 |           0.067 |                         0.029 |                    -0.012 |              0.149 |               0.028 |            0.028 |
| rev_2025q3     | False     |         0.05  |         0.046 |      0.05  |      0.046 |                 0.043 |           0.043 |                        -0.184 |                    -0.042 |              0.158 |               0.041 |            0.041 |
| rev_2026q1     | False     |         0.048 |         0.049 |      0.048 |      0.049 |                 0.05  |           0.05  |                         0.048 |                    -0.05  |              0.138 |               0.044 |            0.044 |
| rev_2026q2     | False     |        -0.004 |        -0.004 |     -0.004 |     -0.004 |                -0.004 |          -0.004 |                        -0.008 |                     0.005 |              0.056 |              -0.01  |           -0.01  |
| rr_2025_12     | False     |        -0.163 |        -0.163 |     -0.163 |     -0.163 |                -0.163 |          -0.163 |                        -0.003 |                    -0.039 |              0.251 |              -0.171 |           -0.171 |

What M1 (pooled) did on the post-t0 targets: CV-chosen lambda, usable nested folds, dropped
features, elasticities in original units, and the predicted weekly growth after t0 split into the
company mean (`rate_base`), mom and the three proxies (other features contribute the rest):

| target         | vintage    | t0         |    rho |   gap_weeks |       lam |   n_folds | feats_dropped                                                                 |   beta_mom |   beta_or_spend |   beta_log_gh_share |   beta_log_ramp_adoption |   rate_base |   contrib_mom |   contrib_or_spend |   contrib_log_gh_share |   contrib_log_ramp_adoption |   rate_hat |   log_err |
|:---------------|:-----------|:-----------|-------:|------------:|----------:|----------:|:------------------------------------------------------------------------------|-----------:|----------------:|--------------------:|-------------------------:|------------:|--------------:|-------------------:|-----------------------:|----------------------------:|-----------:|----------:|
| rr_2026_02     | 2026-02-11 | 2025-10-13 | 0.7291 |       15    |   100     |         0 | openrouter_log_tokens;openrouter_share;vercel_spend_share;pypi_level;or_spend |     0.3738 |        nan      |              0.6358 |                   0.3021 |      0.0406 |       -0.0085 |           nan      |                 0.0223 |                     -0.0016 |     0.0291 |    0.249  |
| rr_2026_03     | 2026-03-02 | 2026-01-26 | 0.7286 |        2.43 |  1000     |         1 | openrouter_log_tokens;openrouter_share;vercel_spend_share;pypi_level;or_spend |     0.021  |        nan      |              0.871  |                   0.7942 |      0.0418 |       -0      |           nan      |                -0.0201 |                     -0.0087 |     0.0187 |    0.2707 |
| rr_2026_04     | 2026-04-05 | 2026-02-15 | 0.8814 |        4.57 |     1e+06 |         2 | openrouter_log_tokens;openrouter_share;vercel_spend_share;pypi_level;or_spend |     0      |        nan      |              0.9998 |                   0.9995 |      0.0438 |        0      |           nan      |                -0.0036 |                      0.0138 |     0.054  |    0.2459 |
| rr_2026_05     | 2026-05-27 | 2026-03-20 | 0.9103 |        7.57 | 10000     |         2 | openrouter_log_tokens;openrouter_share;vercel_spend_share;pypi_level;or_spend |    -0.0016 |        nan      |              0.9817 |                   0.9768 |      0.0523 |       -0.0001 |           nan      |                -0.0125 |                      0.0148 |     0.0541 |    0.0605 |
| rev_2026q3     | 2026-11-01 | 2026-07-14 | 0.9197 |       11.14 |     1e+06 |         2 | vercel_spend_share;pypi_level                                                 |     0.0001 |          0.9998 |              0.9996 |                   1      |      0.0596 |        0      |            -0.1226 |                -0.0436 |                     -0.0098 |    -0.1163 |    0.4496 |
| rr_2026_09_s1a | 2026-11-09 | 2026-07-14 | 0.9196 |        9.14 |     1e+06 |         2 | vercel_spend_share;pypi_level                                                 |     0.0001 |          0.9998 |              0.9996 |                   1      |      0.0596 |        0      |            -0.1343 |                -0.043  |                     -0.0149 |    -0.1326 |    1.11   |
| rr_2026_10_s1a | 2026-11-09 | 2026-07-14 | 0.9196 |       13.57 |     1e+06 |         2 | vercel_spend_share;pypi_level                                                 |     0.0001 |          0.9998 |              0.9996 |                   1      |      0.0596 |        0      |            -0.0427 |                -0.0381 |                     -0.0172 |    -0.0384 |    0.5512 |

### Vintage rule: period_end

Scores with bias (mean log error on post-t0 targets; + = forecast too low):

| model                       |   n post-t0 |   mean |err| post-t0 |   bias post-t0 |   n all |   mean |err| all |
|:----------------------------|------------:|---------------------:|---------------:|--------:|-----------------:|
| M1 (pooled)                 |          11 |                0.286 |          0.271 |      11 |            0.286 |
| M2 (pooled)                 |          11 |                0.122 |          0.09  |      11 |            0.122 |
| M1 (own)                    |          11 |                0.232 |          0.204 |      11 |            0.232 |
| M2 (own)                    |          11 |                0.094 |          0.056 |      11 |            0.094 |
| SS claims-only path         |          11 |                0.224 |         -0.092 |      11 |            0.224 |
| A claims only               |          11 |                0.224 |         -0.092 |      11 |            0.224 |
| E OpenRouter tokens x price |          11 |                0.156 |          0.044 |      11 |            0.156 |
| G GitHub new-repo share     |          11 |                0.151 |          0.125 |      11 |            0.151 |
| naive last claim            |          11 |                0.257 |          0.25  |      11 |            0.257 |
| old AUTO (pooled)           |          11 |                0.224 |          0.004 |      11 |            0.224 |
| old AUTO (own)              |          11 |                0.337 |         -0.112 |      11 |            0.337 |

Paired differences in |log error| (`B vs A`: negative = B better; 90% bootstrap interval;
forecast_eval.paired):

| targets   | rule       | comparison                                 |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:----------|:-----------|:-------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| post-t0   | period_end | M1 (pooled) vs E OpenRouter tokens x price |  11 |       0.13  |     0.03  |     0.256 |          3 |       8 |
| post-t0   | period_end | M1 (pooled) vs G GitHub new-repo share     |  11 |       0.135 |    -0.052 |     0.34  |          5 |       6 |
| post-t0   | period_end | M1 (pooled) vs A claims only               |  11 |       0.062 |    -0.028 |     0.174 |          5 |       6 |
| post-t0   | period_end | M1 (pooled) vs old AUTO (pooled)           |  11 |       0.061 |    -0.047 |     0.188 |          6 |       5 |
| post-t0   | period_end | M1 (pooled) vs naive last claim            |  11 |       0.029 |    -0.177 |     0.264 |          8 |       3 |
| post-t0   | period_end | M2 (pooled) vs E OpenRouter tokens x price |  11 |      -0.034 |    -0.109 |     0.036 |          7 |       4 |
| post-t0   | period_end | M2 (pooled) vs G GitHub new-repo share     |  11 |      -0.029 |    -0.121 |     0.051 |          6 |       5 |
| post-t0   | period_end | M2 (pooled) vs A claims only               |  11 |      -0.102 |    -0.187 |    -0.031 |          9 |       2 |
| post-t0   | period_end | M2 (pooled) vs old AUTO (pooled)           |  11 |      -0.102 |    -0.141 |    -0.064 |          9 |       2 |
| post-t0   | period_end | M2 (pooled) vs naive last claim            |  11 |      -0.135 |    -0.228 |    -0.034 |          9 |       2 |
| post-t0   | period_end | M1 (pooled) vs M1 (own)                    |  11 |       0.053 |     0.005 |     0.111 |          4 |       7 |
| post-t0   | period_end | M2 (pooled) vs M2 (own)                    |  11 |       0.028 |     0.004 |     0.056 |          5 |       6 |
| post-t0   | period_end | M2 (pooled) vs M1 (pooled)                 |  11 |      -0.164 |    -0.313 |    -0.037 |          7 |       4 |
| post-t0   | period_end | M2 (pooled) vs SS claims-only path         |  11 |      -0.102 |    -0.187 |    -0.03  |          9 |       2 |
| all       | period_end | M1 (pooled) vs E OpenRouter tokens x price |  11 |       0.13  |     0.03  |     0.256 |          3 |       8 |
| all       | period_end | M1 (pooled) vs G GitHub new-repo share     |  11 |       0.135 |    -0.052 |     0.34  |          5 |       6 |
| all       | period_end | M1 (pooled) vs A claims only               |  11 |       0.062 |    -0.028 |     0.174 |          5 |       6 |
| all       | period_end | M1 (pooled) vs old AUTO (pooled)           |  11 |       0.061 |    -0.047 |     0.188 |          6 |       5 |
| all       | period_end | M1 (pooled) vs naive last claim            |  11 |       0.029 |    -0.177 |     0.264 |          8 |       3 |
| all       | period_end | M2 (pooled) vs E OpenRouter tokens x price |  11 |      -0.034 |    -0.109 |     0.036 |          7 |       4 |
| all       | period_end | M2 (pooled) vs G GitHub new-repo share     |  11 |      -0.029 |    -0.121 |     0.051 |          6 |       5 |
| all       | period_end | M2 (pooled) vs A claims only               |  11 |      -0.102 |    -0.187 |    -0.031 |          9 |       2 |
| all       | period_end | M2 (pooled) vs old AUTO (pooled)           |  11 |      -0.102 |    -0.141 |    -0.064 |          9 |       2 |
| all       | period_end | M2 (pooled) vs naive last claim            |  11 |      -0.135 |    -0.228 |    -0.034 |          9 |       2 |
| all       | period_end | M1 (pooled) vs M1 (own)                    |  11 |       0.053 |     0.005 |     0.111 |          4 |       7 |
| all       | period_end | M2 (pooled) vs M2 (own)                    |  11 |       0.028 |     0.004 |     0.056 |          5 |       6 |
| all       | period_end | M2 (pooled) vs M1 (pooled)                 |  11 |      -0.164 |    -0.313 |    -0.037 |          7 |       4 |
| all       | period_end | M2 (pooled) vs SS claims-only path         |  11 |      -0.102 |    -0.187 |    -0.03  |          9 |       2 |

Per-target log error (+ = forecast too low):

| target         | post_t0   |   M1 (pooled) |   M2 (pooled) |   M1 (own) |   M2 (own) |   SS claims-only path |   A claims only |   E OpenRouter tokens x price |   G GitHub new-repo share |   naive last claim |   old AUTO (pooled) |   old AUTO (own) |
|:---------------|:----------|--------------:|--------------:|-----------:|-----------:|----------------------:|----------------:|------------------------------:|--------------------------:|-------------------:|--------------------:|-----------------:|
| rev_2025fy     | True      |        -0.031 |        -0.032 |     -0.049 |     -0.041 |                -0.034 |          -0.034 |                        -0.04  |                     0.127 |              0.145 |              -0.017 |           -0.017 |
| rev_2026q1     | True      |         0.06  |         0.019 |      0.06  |      0.019 |                -0.021 |          -0.021 |                        -0.04  |                     0.118 |              0.237 |               0.088 |            0.139 |
| rev_2026q2     | True      |        -0.021 |        -0.07  |     -0.022 |     -0.071 |                -0.119 |          -0.12  |                        -0.127 |                     0.558 |              0.115 |              -0.215 |           -0.174 |
| rev_2026q3     | True      |         0.457 |         0.066 |      0.342 |      0.009 |                -0.324 |          -0.324 |                         0.234 |                    -0.012 |             -0.038 |              -0.267 |           -0.411 |
| rr_2025_12     | True      |        -0.026 |         0.007 |     -0.086 |     -0.023 |                 0.04  |           0.04  |                        -0.003 |                     0.101 |              0.251 |               0.065 |            0.065 |
| rr_2026_02     | True      |         0.249 |         0.306 |      0.103 |      0.233 |                 0.363 |           0.363 |                         0.097 |                     0.113 |              0.693 |               0.359 |            0.36  |
| rr_2026_03     | True      |         0.271 |         0.244 |      0.213 |      0.215 |                 0.217 |           0.217 |                         0.141 |                     0.211 |              0.305 |               0.238 |            0.214 |
| rr_2026_04     | True      |         0.246 |         0.177 |      0.246 |      0.177 |                 0.108 |           0.108 |                         0.081 |                     0.192 |              0.457 |               0.379 |            0.458 |
| rr_2026_05     | True      |         0.061 |        -0.073 |      0.057 |     -0.075 |                -0.206 |          -0.206 |                         0.171 |                     0.098 |              0.449 |               0.128 |           -0.039 |
| rr_2026_09_s1a | True      |         1.149 |         0.327 |      0.805 |      0.155 |                -0.494 |          -0.494 |                         0.376 |                    -0.124 |              0     |              -0.526 |           -0.703 |
| rr_2026_10_s1a | True      |         0.572 |         0.018 |      0.573 |      0.018 |                -0.537 |          -0.537 |                        -0.401 |                    -0.005 |              0.132 |              -0.184 |           -1.121 |

What M1 (pooled) did on the post-t0 targets: CV-chosen lambda, usable nested folds, dropped
features, elasticities in original units, and the predicted weekly growth after t0 split into the
company mean (`rate_base`), mom and the three proxies (other features contribute the rest):

| target         | vintage    | t0         |    rho |   gap_weeks |       lam |   n_folds | feats_dropped                                                                 |   beta_mom |   beta_or_spend |   beta_log_gh_share |   beta_log_ramp_adoption |   rate_base |   contrib_mom |   contrib_or_spend |   contrib_log_gh_share |   contrib_log_ramp_adoption |   rate_hat |   log_err |
|:---------------|:-----------|:-----------|-------:|------------:|----------:|----------:|:------------------------------------------------------------------------------|-----------:|----------------:|--------------------:|-------------------------:|------------:|--------------:|-------------------:|-----------------------:|----------------------------:|-----------:|----------:|
| rr_2025_12     | 2025-12-28 | 2025-10-13 | 0.7291 |        8.86 |   100     |         0 | openrouter_log_tokens;openrouter_share;vercel_spend_share;pypi_level;or_spend |     0.3582 |        nan      |              0.6362 |                   0.3087 |      0.0406 |       -0.009  |           nan      |                 0.0114 |                     -0.002  |     0.0304 |   -0.0256 |
| rev_2025fy     | 2025-12-31 | 2025-10-13 | 0.7291 |       11.29 |   100     |         0 | openrouter_log_tokens;openrouter_share;vercel_spend_share;pypi_level;or_spend |     0.3582 |        nan      |              0.6362 |                   0.3087 |      0.0406 |       -0.0094 |           nan      |                 0.0183 |                     -0.001  |     0.0352 |   -0.0306 |
| rr_2026_02     | 2026-02-09 | 2025-10-13 | 0.7291 |       15    |   100     |         0 | openrouter_log_tokens;openrouter_share;vercel_spend_share;pypi_level;or_spend |     0.3738 |        nan      |              0.6358 |                   0.3021 |      0.0406 |       -0.0085 |           nan      |                 0.0223 |                     -0.0016 |     0.0291 |    0.249  |
| rr_2026_03     | 2026-02-22 | 2026-01-26 | 0.7286 |        2.43 |  1000     |         1 | openrouter_log_tokens;openrouter_share;vercel_spend_share;pypi_level;or_spend |     0.021  |        nan      |              0.871  |                   0.7942 |      0.0418 |       -0      |           nan      |                -0.0201 |                     -0.0087 |     0.0187 |    0.2707 |
| rr_2026_04     | 2026-03-20 | 2026-02-15 | 0.8814 |        4.57 |     1e+06 |         2 | openrouter_log_tokens;openrouter_share;vercel_spend_share;pypi_level;or_spend |     0      |        nan      |              0.9998 |                   0.9997 |      0.0438 |        0      |           nan      |                -0.0036 |                      0.0138 |     0.054  |    0.2459 |
| rev_2026q1     | 2026-03-31 | 2026-02-15 | 0.8814 |        6.29 |     1e+06 |         2 | openrouter_log_tokens;openrouter_share;vercel_spend_share;pypi_level;or_spend |     0      |        nan      |              0.9998 |                   0.9995 |      0.0438 |        0      |           nan      |                 0.0014 |                      0.0138 |     0.059  |    0.06   |
| rr_2026_05     | 2026-05-15 | 2026-03-20 | 0.9103 |        7.57 | 10000     |         2 | openrouter_log_tokens;openrouter_share;vercel_spend_share;pypi_level;or_spend |    -0.001  |        nan      |              0.9816 |                   0.9767 |      0.0523 |       -0      |           nan      |                -0.0125 |                      0.0148 |     0.0541 |    0.0606 |
| rev_2026q2     | 2026-06-30 | 2026-05-11 | 0.8988 |        7.14 |     1e+06 |         2 | openrouter_log_tokens;openrouter_share;vercel_spend_share;pypi_level;or_spend |    -0      |        nan      |              0.9998 |                   0.9999 |      0.0583 |       -0      |           nan      |                -0.0397 |                      0.0207 |     0.0393 |   -0.0214 |
| rr_2026_09_s1a | 2026-09-30 | 2026-07-14 | 0.9197 |        9.14 |     1e+06 |         2 | vercel_spend_share;pypi_level                                                 |     0.0001 |          0.9998 |              0.9997 |                   1      |      0.0596 |        0      |            -0.1363 |                -0.0431 |                     -0.0149 |    -0.1347 |    1.1489 |
| rev_2026q3     | 2026-09-30 | 2026-07-14 | 0.9197 |       11.14 |     1e+06 |         2 | vercel_spend_share;pypi_level                                                 |     0.0001 |          0.9998 |              0.9997 |                   1      |      0.0596 |        0      |            -0.1246 |                -0.0437 |                     -0.0098 |    -0.1184 |    0.4571 |
| rr_2026_10_s1a | 2026-10-31 | 2026-07-14 | 0.9197 |       13.57 |     1e+06 |         2 | vercel_spend_share;pypi_level                                                 |     0.0001 |          0.9998 |              0.9996 |                   1      |      0.0596 |        0      |            -0.0427 |                -0.0382 |                     -0.0172 |    -0.0385 |    0.5718 |

## Ablations (pre-declared diagnostics, not candidates)

Each changes one thing in M1 (see the specification, item 8). Same common targets as the headline.

### publication

| model                       |   n post-t0 |   mean |err| post-t0 |   bias post-t0 |   n all |   mean |err| all |
|:----------------------------|------------:|---------------------:|---------------:|--------:|-----------------:|
| M1 (pooled)                 |           7 |                0.42  |          0.42  |      14 |            0.245 |
| M1 prior 0 (pooled)         |           7 |                0.233 |         -0.016 |      14 |            0.152 |
| M1 inner CV (pooled)        |           7 |                0.229 |          0.043 |      14 |            0.15  |
| M1 old prep (pooled)        |           7 |                0.197 |         -0.017 |      14 |            0.134 |
| M1 rho 0.5 (pooled)         |           7 |                0.452 |          0.452 |      14 |            0.259 |
| M1 prior 1/3 (pooled)       |           7 |                0.176 |          0.049 |      14 |            0.123 |
| M1 (own)                    |           7 |                0.328 |          0.328 |      14 |            0.199 |
| M1 prior 0 (own)            |           7 |                0.251 |         -0.053 |      14 |            0.16  |
| M1 inner CV (own)           |           7 |                0.397 |          0.397 |      14 |            0.234 |
| M1 old prep (own)           |           7 |                0.178 |          0.033 |      14 |            0.124 |
| M1 rho 0.5 (own)            |           7 |                0.349 |          0.343 |      14 |            0.208 |
| M1 prior 1/3 (own)          |           7 |                0.182 |          0.009 |      14 |            0.126 |
| SS claims-only path         |           7 |                0.347 |         -0.151 |      14 |            0.207 |
| SS claims-only path rho 0.5 |           7 |                0.36  |         -0.125 |      14 |            0.211 |
| A claims only               |           7 |                0.347 |         -0.151 |      14 |            0.207 |

Paired, ablation vs M1 (`ablation vs M1`: negative = the ablation is better):

| targets   | rule        | comparison                                         |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:----------|:------------|:---------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| post-t0   | publication | M1 prior 0 (pooled) vs M1 (pooled)                 |   7 |      -0.186 |    -0.372 |    -0.043 |          5 |       2 |
| post-t0   | publication | M1 inner CV (pooled) vs M1 (pooled)                |   7 |      -0.19  |    -0.396 |    -0.011 |          4 |       3 |
| post-t0   | publication | M1 old prep (pooled) vs M1 (pooled)                |   7 |      -0.222 |    -0.434 |    -0.032 |          4 |       3 |
| post-t0   | publication | M1 rho 0.5 (pooled) vs M1 (pooled)                 |   7 |       0.032 |     0.011 |     0.054 |          2 |       5 |
| post-t0   | publication | M1 prior 1/3 (pooled) vs M1 (pooled)               |   7 |      -0.244 |    -0.497 |    -0.051 |          5 |       2 |
| post-t0   | publication | M1 prior 0 (own) vs M1 (own)                       |   7 |      -0.078 |    -0.181 |     0.003 |          4 |       3 |
| post-t0   | publication | M1 inner CV (own) vs M1 (own)                      |   7 |       0.069 |     0     |     0.198 |          0 |       2 |
| post-t0   | publication | M1 old prep (own) vs M1 (own)                      |   7 |      -0.151 |    -0.341 |     0.038 |          4 |       3 |
| post-t0   | publication | M1 rho 0.5 (own) vs M1 (own)                       |   7 |       0.021 |    -0.003 |     0.046 |          3 |       4 |
| post-t0   | publication | M1 prior 1/3 (own) vs M1 (own)                     |   7 |      -0.147 |    -0.293 |    -0.021 |          4 |       3 |
| post-t0   | publication | SS claims-only path rho 0.5 vs SS claims-only path |   7 |       0.013 |    -0.027 |     0.053 |          3 |       4 |
| all       | publication | M1 prior 0 (pooled) vs M1 (pooled)                 |  14 |      -0.093 |    -0.193 |    -0.016 |          5 |       2 |
| all       | publication | M1 inner CV (pooled) vs M1 (pooled)                |  14 |      -0.095 |    -0.209 |    -0.003 |          4 |       3 |
| all       | publication | M1 old prep (pooled) vs M1 (pooled)                |  14 |      -0.111 |    -0.23  |    -0.011 |          4 |       3 |
| all       | publication | M1 rho 0.5 (pooled) vs M1 (pooled)                 |  14 |       0.015 |     0.001 |     0.029 |          5 |       9 |
| all       | publication | M1 prior 1/3 (pooled) vs M1 (pooled)               |  14 |      -0.122 |    -0.257 |    -0.017 |          5 |       2 |
| all       | publication | M1 prior 0 (own) vs M1 (own)                       |  14 |      -0.039 |    -0.093 |     0.002 |          4 |       3 |
| all       | publication | M1 inner CV (own) vs M1 (own)                      |  14 |       0.034 |     0     |     0.099 |          0 |       2 |
| all       | publication | M1 old prep (own) vs M1 (own)                      |  14 |      -0.075 |    -0.179 |     0.02  |          4 |       3 |
| all       | publication | M1 rho 0.5 (own) vs M1 (own)                       |  14 |       0.009 |    -0.005 |     0.023 |          6 |       8 |
| all       | publication | M1 prior 1/3 (own) vs M1 (own)                     |  14 |      -0.073 |    -0.155 |    -0.01  |          4 |       3 |
| all       | publication | SS claims-only path rho 0.5 vs SS claims-only path |  14 |       0.004 |    -0.017 |     0.026 |          6 |       8 |

### period_end

| model                       |   n post-t0 |   mean |err| post-t0 |   bias post-t0 |   n all |   mean |err| all |
|:----------------------------|------------:|---------------------:|---------------:|--------:|-----------------:|
| M1 (pooled)                 |          11 |                0.286 |          0.271 |      11 |            0.286 |
| M1 prior 0 (pooled)         |          11 |                0.175 |         -0.021 |      11 |            0.175 |
| M1 inner CV (pooled)        |          11 |                0.198 |          0.158 |      11 |            0.198 |
| M1 old prep (pooled)        |          11 |                0.158 |         -0.025 |      11 |            0.158 |
| M1 rho 0.5 (pooled)         |          11 |                0.314 |          0.291 |      11 |            0.314 |
| M1 prior 1/3 (pooled)       |          11 |                0.126 |          0.031 |      11 |            0.126 |
| M1 (own)                    |          11 |                0.232 |          0.204 |      11 |            0.232 |
| M1 prior 0 (own)            |          11 |                0.182 |         -0.04  |      11 |            0.182 |
| M1 inner CV (own)           |          11 |                0.27  |          0.266 |      11 |            0.27  |
| M1 old prep (own)           |          11 |                0.144 |          0.029 |      11 |            0.144 |
| M1 rho 0.5 (own)            |          11 |                0.253 |          0.212 |      11 |            0.253 |
| M1 prior 1/3 (own)          |          11 |                0.136 |         -0.001 |      11 |            0.136 |
| SS claims-only path         |          11 |                0.224 |         -0.092 |      11 |            0.224 |
| SS claims-only path rho 0.5 |          11 |                0.238 |         -0.08  |      11 |            0.238 |
| A claims only               |          11 |                0.224 |         -0.092 |      11 |            0.224 |

Paired, ablation vs M1 (`ablation vs M1`: negative = the ablation is better):

| targets   | rule       | comparison                                         |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:----------|:-----------|:---------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| post-t0   | period_end | M1 prior 0 (pooled) vs M1 (pooled)                 |  11 |      -0.11  |    -0.24  |    -0.011 |          6 |       5 |
| post-t0   | period_end | M1 inner CV (pooled) vs M1 (pooled)                |  11 |      -0.088 |    -0.192 |     0.001 |          5 |       6 |
| post-t0   | period_end | M1 old prep (pooled) vs M1 (pooled)                |  11 |      -0.128 |    -0.277 |    -0.003 |          6 |       5 |
| post-t0   | period_end | M1 rho 0.5 (pooled) vs M1 (pooled)                 |  11 |       0.028 |     0.014 |     0.043 |          2 |       9 |
| post-t0   | period_end | M1 prior 1/3 (pooled) vs M1 (pooled)               |  11 |      -0.159 |    -0.343 |    -0.02  |          6 |       5 |
| post-t0   | period_end | M1 prior 0 (own) vs M1 (own)                       |  11 |      -0.05  |    -0.129 |     0.01  |          6 |       5 |
| post-t0   | period_end | M1 inner CV (own) vs M1 (own)                      |  11 |       0.038 |    -0.011 |     0.121 |          2 |       4 |
| post-t0   | period_end | M1 old prep (own) vs M1 (own)                      |  11 |      -0.088 |    -0.232 |     0.046 |          6 |       5 |
| post-t0   | period_end | M1 rho 0.5 (own) vs M1 (own)                       |  11 |       0.02  |     0.005 |     0.036 |          4 |       7 |
| post-t0   | period_end | M1 prior 1/3 (own) vs M1 (own)                     |  11 |      -0.096 |    -0.209 |    -0.008 |          6 |       5 |
| post-t0   | period_end | SS claims-only path rho 0.5 vs SS claims-only path |  11 |       0.014 |    -0.025 |     0.052 |          3 |       8 |
| all       | period_end | M1 prior 0 (pooled) vs M1 (pooled)                 |  11 |      -0.11  |    -0.24  |    -0.011 |          6 |       5 |
| all       | period_end | M1 inner CV (pooled) vs M1 (pooled)                |  11 |      -0.088 |    -0.192 |     0.001 |          5 |       6 |
| all       | period_end | M1 old prep (pooled) vs M1 (pooled)                |  11 |      -0.128 |    -0.277 |    -0.003 |          6 |       5 |
| all       | period_end | M1 rho 0.5 (pooled) vs M1 (pooled)                 |  11 |       0.028 |     0.014 |     0.043 |          2 |       9 |
| all       | period_end | M1 prior 1/3 (pooled) vs M1 (pooled)               |  11 |      -0.159 |    -0.343 |    -0.02  |          6 |       5 |
| all       | period_end | M1 prior 0 (own) vs M1 (own)                       |  11 |      -0.05  |    -0.129 |     0.01  |          6 |       5 |
| all       | period_end | M1 inner CV (own) vs M1 (own)                      |  11 |       0.038 |    -0.011 |     0.121 |          2 |       4 |
| all       | period_end | M1 old prep (own) vs M1 (own)                      |  11 |      -0.088 |    -0.232 |     0.046 |          6 |       5 |
| all       | period_end | M1 rho 0.5 (own) vs M1 (own)                       |  11 |       0.02  |     0.005 |     0.036 |          4 |       7 |
| all       | period_end | M1 prior 1/3 (own) vs M1 (own)                     |  11 |      -0.096 |    -0.209 |    -0.008 |          6 |       5 |
| all       | period_end | SS claims-only path rho 0.5 vs SS claims-only path |  11 |       0.014 |    -0.025 |     0.052 |          3 |       8 |

## Second test set: OpenAI claims

OpenAI's included, non-forecast point total claims (all tiers); pooled training only. References: the
claims-only path from the same label run, the panel's naive last claim (both at OpenAI's profiled rho) and
the old zoo AUTO (pooled, g_spec; rho 0.5).

### publication

| model                    |   n post-t0 |   mean |err| post-t0 |   bias post-t0 |   n all |   mean |err| all |
|:-------------------------|------------:|---------------------:|---------------:|--------:|-----------------:|
| M1 (pooled)              |          10 |                0.062 |          0.024 |      15 |            0.067 |
| M2 (pooled)              |          10 |                0.081 |          0.014 |      15 |            0.079 |
| SS claims-only path      |          10 |                0.11  |          0.004 |      15 |            0.098 |
| naive last claim (panel) |          10 |                0.186 |          0.183 |      15 |            0.159 |
| old AUTO (pooled)        |          10 |                0.119 |         -0.025 |      15 |            0.106 |
| M1 prior 0 (pooled)      |          10 |                0.078 |         -0.019 |      15 |            0.077 |
| M1 prior 1/3 (pooled)    |          10 |                0.068 |         -0.039 |      15 |            0.071 |

| targets   | rule        | comparison                              |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:----------|:------------|:----------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| post-t0   | publication | M1 (pooled) vs SS claims-only path      |  10 |      -0.048 |    -0.09  |    -0.01  |          8 |       2 |
| post-t0   | publication | M1 (pooled) vs naive last claim (panel) |  10 |      -0.124 |    -0.206 |    -0.052 |          7 |       3 |
| post-t0   | publication | M1 (pooled) vs old AUTO (pooled)        |  10 |      -0.057 |    -0.115 |    -0.012 |          8 |       2 |
| post-t0   | publication | M2 (pooled) vs SS claims-only path      |  10 |      -0.029 |    -0.055 |    -0.007 |          8 |       2 |
| post-t0   | publication | M2 (pooled) vs naive last claim (panel) |  10 |      -0.105 |    -0.179 |    -0.041 |          8 |       2 |
| post-t0   | publication | M2 (pooled) vs old AUTO (pooled)        |  10 |      -0.038 |    -0.083 |    -0.004 |          8 |       2 |
| all       | publication | M1 (pooled) vs SS claims-only path      |  15 |      -0.032 |    -0.062 |    -0.006 |          9 |       5 |
| all       | publication | M1 (pooled) vs naive last claim (panel) |  15 |      -0.092 |    -0.152 |    -0.04  |         10 |       5 |
| all       | publication | M1 (pooled) vs old AUTO (pooled)        |  15 |      -0.039 |    -0.08  |    -0.009 |         13 |       2 |
| all       | publication | M2 (pooled) vs SS claims-only path      |  15 |      -0.019 |    -0.038 |    -0.004 |          9 |       5 |
| all       | publication | M2 (pooled) vs naive last claim (panel) |  15 |      -0.08  |    -0.133 |    -0.034 |         11 |       4 |
| all       | publication | M2 (pooled) vs old AUTO (pooled)        |  15 |      -0.027 |    -0.059 |    -0.004 |         13 |       2 |

| target                 | post_t0   |   M1 (pooled) |   M2 (pooled) |   SS claims-only path |   naive last claim (panel) |   old AUTO (pooled) |
|:-----------------------|:----------|--------------:|--------------:|----------------------:|---------------------------:|--------------------:|
| oa_rev_2025fy_cnbc     | True      |         0.003 |        -0.019 |                -0.041 |                      0.221 |              -0.043 |
| oa_rev_2026q1_ti       | True      |        -0.048 |        -0.049 |                -0.05  |                      0.015 |              -0.045 |
| oa_rev_2026q3_ti       | True      |        -0.034 |        -0.034 |                -0.035 |                     -0.02  |              -0.037 |
| oa_rr_2025_07_ti       | True      |         0.023 |         0.038 |                 0.053 |                      0.182 |               0.046 |
| oa_rr_2025_08_epoch    | True      |        -0.042 |        -0.039 |                -0.037 |                      0.08  |              -0.061 |
| oa_rr_2025_12_ti       | True      |        -0.007 |        -0.056 |                -0.106 |                      0.498 |              -0.137 |
| oa_rr_2026_03_official | True      |         0.039 |        -0.112 |                -0.263 |                      0.115 |              -0.369 |
| oa_rr_2026_04_semafor  | True      |        -0.061 |        -0.027 |                 0.007 |                      0.041 |              -0.027 |
| oa_rr_2026_08_bbg      | True      |         0.26  |         0.325 |                 0.39  |                      0.47  |               0.292 |
| oa_rr_2026_09_ft       | True      |         0.106 |         0.112 |                 0.119 |                      0.223 |               0.132 |
| oa_rev_2025_12_ft      | False     |         0.079 |         0.079 |                 0.079 |                      0.14  |               0.083 |
| oa_rev_2025fy_audit    | False     |         0.017 |         0.017 |                 0.017 |                      0.085 |               0.018 |
| oa_rev_2025h1_ti       | False     |         0.034 |         0.033 |                 0.032 |                      0.084 |               0.035 |
| oa_rev_2026q2_wsj      | False     |        -0.12  |        -0.121 |                -0.121 |                     -0.087 |              -0.13  |
| oa_rr_2026_06_derived  | False     |        -0.124 |        -0.124 |                -0.124 |                      0.12  |              -0.133 |

### period_end

| model                    |   n post-t0 |   mean |err| post-t0 |   bias post-t0 |   n all |   mean |err| all |
|:-------------------------|------------:|---------------------:|---------------:|--------:|-----------------:|
| M1 (pooled)              |          15 |                0.063 |          0.019 |      15 |            0.063 |
| M2 (pooled)              |          15 |                0.063 |          0.028 |      15 |            0.063 |
| SS claims-only path      |          15 |                0.079 |          0.037 |      15 |            0.079 |
| naive last claim (panel) |          15 |                0.199 |          0.199 |      15 |            0.199 |
| old AUTO (pooled)        |          15 |                0.085 |          0.001 |      15 |            0.085 |
| M1 prior 0 (pooled)      |          15 |                0.074 |         -0.012 |      15 |            0.074 |
| M1 prior 1/3 (pooled)    |          15 |                0.077 |         -0.036 |      15 |            0.077 |

| targets   | rule       | comparison                              |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:----------|:-----------|:----------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| post-t0   | period_end | M1 (pooled) vs SS claims-only path      |  15 |      -0.016 |    -0.05  |     0.013 |          8 |       7 |
| post-t0   | period_end | M1 (pooled) vs naive last claim (panel) |  15 |      -0.136 |    -0.216 |    -0.064 |         11 |       4 |
| post-t0   | period_end | M1 (pooled) vs old AUTO (pooled)        |  15 |      -0.022 |    -0.066 |     0.015 |         10 |       5 |
| post-t0   | period_end | M2 (pooled) vs SS claims-only path      |  15 |      -0.016 |    -0.035 |    -0.002 |         10 |       5 |
| post-t0   | period_end | M2 (pooled) vs naive last claim (panel) |  15 |      -0.136 |    -0.215 |    -0.067 |         13 |       2 |
| post-t0   | period_end | M2 (pooled) vs old AUTO (pooled)        |  15 |      -0.023 |    -0.056 |     0.003 |          9 |       6 |
| all       | period_end | M1 (pooled) vs SS claims-only path      |  15 |      -0.016 |    -0.05  |     0.013 |          8 |       7 |
| all       | period_end | M1 (pooled) vs naive last claim (panel) |  15 |      -0.136 |    -0.216 |    -0.064 |         11 |       4 |
| all       | period_end | M1 (pooled) vs old AUTO (pooled)        |  15 |      -0.022 |    -0.066 |     0.015 |         10 |       5 |
| all       | period_end | M2 (pooled) vs SS claims-only path      |  15 |      -0.016 |    -0.035 |    -0.002 |         10 |       5 |
| all       | period_end | M2 (pooled) vs naive last claim (panel) |  15 |      -0.136 |    -0.215 |    -0.067 |         13 |       2 |
| all       | period_end | M2 (pooled) vs old AUTO (pooled)        |  15 |      -0.023 |    -0.056 |     0.003 |          9 |       6 |

| target                 | post_t0   |   M1 (pooled) |   M2 (pooled) |   SS claims-only path |   naive last claim (panel) |   old AUTO (pooled) |
|:-----------------------|:----------|--------------:|--------------:|----------------------:|---------------------------:|--------------------:|
| oa_rev_2025_12_ft      | True      |         0.026 |         0.028 |                 0.03  |                      0.594 |              -0.041 |
| oa_rev_2025fy_audit    | True      |        -0.001 |        -0     |                 0.001 |                      0.23  |              -0.023 |
| oa_rev_2025fy_cnbc     | True      |         0.001 |         0.002 |                 0.003 |                      0.232 |              -0.021 |
| oa_rev_2025h1_ti       | True      |         0.136 |         0.134 |                 0.133 |                      0.093 |               0.129 |
| oa_rev_2026q1_ti       | True      |        -0.057 |        -0.054 |                -0.05  |                      0.015 |              -0.048 |
| oa_rev_2026q2_wsj      | True      |        -0.015 |         0.008 |                 0.03  |                      0.073 |              -0.022 |
| oa_rev_2026q3_ti       | True      |         0.003 |         0.003 |                 0.003 |                      0.021 |               0.013 |
| oa_rr_2025_07_ti       | True      |         0.023 |         0.038 |                 0.053 |                      0.182 |               0.046 |
| oa_rr_2025_08_epoch    | True      |         0.023 |         0.033 |                 0.042 |                      0.08  |               0.013 |
| oa_rr_2025_12_ti       | True      |        -0.016 |        -0.004 |                 0.008 |                      0.498 |              -0.09  |
| oa_rr_2026_03_official | True      |         0.039 |        -0.112 |                -0.263 |                      0.115 |              -0.369 |
| oa_rr_2026_04_semafor  | True      |        -0.061 |        -0.027 |                 0.007 |                      0.041 |              -0.021 |
| oa_rr_2026_06_derived  | True      |        -0.18  |        -0.063 |                 0.054 |                      0.12  |               0.02  |
| oa_rr_2026_08_bbg      | True      |         0.26  |         0.325 |                 0.39  |                      0.47  |               0.292 |
| oa_rr_2026_09_ft       | True      |         0.106 |         0.112 |                 0.119 |                      0.223 |               0.132 |

## Run notes

- scikit-learn 1.9.1; numpy 2.5.3; 4 worker processes, every fit single-threaded.
- lambda chosen by the nested CV for M1 (count of targets):

|                                        |   10.0 |   100.0 |   1000.0 |   10000.0 |   1000000.0 |
|:---------------------------------------|-------:|--------:|---------:|----------:|------------:|
| ('anthropic', 'period_end', 'own')     |      0 |       4 |        0 |         1 |           6 |
| ('anthropic', 'period_end', 'pooled')  |      0 |       3 |        1 |         1 |           6 |
| ('anthropic', 'publication', 'own')    |      0 |       3 |        0 |         1 |          10 |
| ('anthropic', 'publication', 'pooled') |      0 |       1 |        2 |         1 |          10 |
| ('openai', 'period_end', 'pooled')     |      2 |       7 |        5 |         0 |           1 |
| ('openai', 'publication', 'pooled')    |      4 |       5 |        5 |         0 |           1 |

- publication: 30 targets with a row; summed job time 10.6 CPU-minutes.
- period_end: 27 targets with a row; summed job time 8.9 CPU-minutes.
- Crossing alignment check: the claims-only path from the label run (profiled rho) against A's log error on the same targets: publication: n=14, mean |SS - A| = 0.0000, max 0.0001; period_end: n=11, mean |SS - A| = 0.0000, max 0.0000.

