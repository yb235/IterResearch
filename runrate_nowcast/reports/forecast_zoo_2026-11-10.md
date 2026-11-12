# Model zoo on the pooled two-company panel: backtest (2026-11-10 data)

Protocol: the docstring of `model_zoo.py`, fixed before the first run. Labels, training pairs, forecasts
and targets are those of `panel_model.py`. 18 members (trend, ridge, lasso, enet, pca, pls, bayes_ridge, ard, kernel_ridge, gp, knn, rf, extra_trees, hgb, lightgbm, xgboost, mlp, midas);
hyperparameters by purged time-ordered CV inside the training pairs public at each vintage.
**AUTO** (declared in advance as the headline) = at each vintage, the member with the lowest inner-CV
error on training pairs. **stack** = non-negative weights summing to one on the members' out-of-fold
predictions. `pooled` = Anthropic + OpenAI pairs; `own` = Anthropic only (pre-declared control).
Errors are log(actual / forecast); negative = forecast too high. Filing rows are never inputs.
Feature sets in this report: g_spec.

## Anthropic targets (g_spec)

### Headline: mean |log error| on common Anthropic targets (g_spec)

Common targets = those forecast by AUTO, stack (both pools), A, E, G and the naive last claim.
AUTO (pooled) is the pre-declared headline model.

| model                         |   publication: n_common |   publication: mae_log_common |   publication: mae_log_h2_2026 |   period_end: n_common |   period_end: mae_log_common |   period_end: mae_log_h2_2026 |
|:------------------------------|------------------------:|------------------------------:|-------------------------------:|-----------------------:|-----------------------------:|------------------------------:|
| AUTO (pooled)                 |                      14 |                         0.15  |                          0.177 |                     11 |                        0.224 |                         0.326 |
| stack (pooled)                |                      14 |                         0.175 |                          0.319 |                     11 |                        0.265 |                         0.508 |
| AUTO (own)                    |                      14 |                         0.273 |                          0.761 |                     11 |                        0.337 |                         0.745 |
| stack (own)                   |                      14 |                         0.274 |                          0.761 |                     11 |                        0.335 |                         0.745 |
| SS claims only (panel labels) |                      14 |                         0.211 |                          0.462 |                     11 |                        0.238 |                         0.357 |
| A claims only                 |                      14 |                         0.207 |                          0.512 |                     11 |                        0.224 |                         0.452 |
| E OpenRouter tokens x price   |                      14 |                         0.105 |                          0.2   |                     11 |                        0.156 |                         0.337 |
| G GitHub new-repo share       |                      14 |                         0.099 |                          0.203 |                     11 |                        0.151 |                         0.047 |
| naive last claim              |                      14 |                         0.221 |                          0.056 |                     11 |                        0.257 |                         0.056 |
| P panel pca (pooled)          |                      14 |                         0.19  |                          0.376 |                     11 |                        0.312 |                         0.662 |
| P panel ridge (pooled)        |                      14 |                         0.137 |                          0.177 |                     11 |                        0.221 |                         0.346 |

### Vintage rule: publication (g_spec)

All models, own targets and common targets (bias: mean log error, + = forecast too low):

| model                         |   n |   mae_log |   n_common |   mae_log_common |   bias_log_common |   mae_log_h2_2026 |
|:------------------------------|----:|----------:|-----------:|-----------------:|------------------:|------------------:|
| AUTO (pooled)                 |  14 |     0.15  |         14 |            0.15  |             0.049 |             0.177 |
| stack (pooled)                |  14 |     0.175 |         14 |            0.175 |             0.013 |             0.319 |
| AUTO (own)                    |  14 |     0.273 |         14 |            0.273 |            -0.084 |             0.761 |
| stack (own)                   |  14 |     0.274 |         14 |            0.274 |            -0.078 |             0.761 |
| SS claims only (panel labels) |  14 |     0.211 |         14 |            0.211 |            -0.057 |             0.462 |
| A claims only                 |  15 |     0.223 |         14 |            0.207 |            -0.065 |             0.512 |
| E OpenRouter tokens x price   |  15 |     0.132 |         14 |            0.105 |             0.036 |             0.2   |
| G GitHub new-repo share       |  15 |     0.1   |         14 |            0.099 |            -0.012 |             0.203 |
| naive last claim              |  15 |     0.262 |         14 |            0.221 |             0.215 |             0.056 |
| P panel pca (pooled)          |  14 |     0.19  |         14 |            0.19  |             0.004 |             0.376 |
| P panel ridge (pooled)        |  14 |     0.137 |         14 |            0.137 |             0.035 |             0.177 |

Every member, mean |log error| on the targets AUTO forecast. The best member is selected after
the fact and therefore optimistic; ranks are among the members only:

| model        |   pooled |   own |   rank_pooled |   rank_own |
|:-------------|---------:|------:|--------------:|-----------:|
| ridge        |    0.137 | 0.215 |             1 |         10 |
| ard          |    0.141 | 0.188 |             2 |          2 |
| bayes_ridge  |    0.144 | 0.182 |             3 |          1 |
| AUTO         |    0.15  | 0.273 |           nan |        nan |
| lightgbm     |    0.158 | 0.195 |             4 |          4 |
| pls          |    0.161 | 0.217 |             5 |         12 |
| xgboost      |    0.166 | 0.196 |             6 |          5 |
| enet         |    0.171 | 0.225 |             7 |         14 |
| rf           |    0.172 | 0.206 |             8 |          6 |
| hgb          |    0.175 | 0.192 |             9 |          3 |
| stack        |    0.175 | 0.274 |           nan |        nan |
| pca          |    0.19  | 0.247 |            10 |         17 |
| mlp          |    0.191 | 0.255 |            11 |         18 |
| midas        |    0.197 | 0.216 |            12 |         11 |
| lasso        |    0.203 | 0.215 |            13 |          9 |
| knn          |    0.207 | 0.241 |            14 |         16 |
| extra_trees  |    0.208 | 0.214 |            15 |          8 |
| gp           |    0.225 | 0.211 |            16 |          7 |
| kernel_ridge |    0.226 | 0.225 |            17 |         13 |
| trend        |    0.239 | 0.24  |            18 |         15 |

Spread of member scores, AUTO's rank, and a leave-one-target-out best-of-zoo (pick the member
with the lowest error on the other targets, score it on the held-out one):

| pool   |   n_targets |   n_members | best member (optimistic)   |   best |   p25 |   median |   p75 |   worst | worst member   |   AUTO |   AUTO rank among members |   stack |   LOO best-of-zoo |
|:-------|------------:|------------:|:---------------------------|-------:|------:|---------:|------:|--------:|:---------------|-------:|--------------------------:|--------:|------------------:|
| pooled |          14 |          18 | ridge                      |  0.137 | 0.162 |    0.183 | 0.206 |   0.239 | trend          |  0.15  |                         4 |   0.175 |             0.166 |
| own    |          14 |          18 | bayes_ridge                |  0.182 | 0.198 |    0.215 | 0.225 |   0.255 | mlp            |  0.273 |                        19 |   0.274 |             0.254 |

AUTO's pick and log error per target:

| target         | vintage    |   gap_weeks | own   | pooled       |   log_err AUTO own |   log_err AUTO pooled |
|:---------------|:-----------|------------:|:------|:-------------|-------------------:|----------------------:|
| rr_2026_02     | 2026-02-11 |       15.43 | trend | pca          |              0.36  |                 0.359 |
| rr_2025_12     | 2026-03-02 |        0    | mlp   | rf           |             -0.171 |                -0.171 |
| rr_2026_03     | 2026-03-02 |        1.86 | mlp   | rf           |              0.214 |                 0.238 |
| rr_2026_04     | 2026-04-05 |        3.71 | pca   | rf           |              0.458 |                 0.379 |
| rr_2026_05     | 2026-05-27 |        8    | mlp   | kernel_ridge |             -0.039 |                 0.128 |
| rev_2025fy     | 2026-10-25 |        0    | mlp   | ridge        |              0.08  |                 0.08  |
| rev_2025q1     | 2026-10-25 |        0    | mlp   | ridge        |              0.098 |                 0.098 |
| rev_2025q2     | 2026-10-25 |        0    | mlp   | ridge        |              0.028 |                 0.028 |
| rev_2025q3     | 2026-10-25 |        0    | mlp   | ridge        |              0.041 |                 0.041 |
| rev_2026q1     | 2026-10-25 |        0    | mlp   | ridge        |              0.044 |                 0.044 |
| rev_2026q2     | 2026-10-25 |        0    | mlp   | ridge        |             -0.01  |                -0.01  |
| rev_2026q3     | 2026-11-01 |       11.14 | mlp   | ridge        |             -0.408 |                -0.041 |
| rr_2026_09_s1a | 2026-11-09 |        9.14 | mlp   | ridge        |             -0.729 |                -0.233 |
| rr_2026_10_s1a | 2026-11-09 |       13.57 | mlp   | ridge        |             -1.145 |                -0.256 |

H2 2026 targets, % error (+ = forecast too high):

| model                         |   rev_2026q3 |   rr_2026_09_s1a |   rr_2026_10_s1a |
|:------------------------------|-------------:|-----------------:|-----------------:|
| AUTO (pooled)                 |          4.2 |             26.3 |             29.1 |
| stack (pooled)                |         17.4 |             46.8 |             51.1 |
| AUTO (own)                    |         50.3 |            107.4 |            214.3 |
| stack (own)                   |         50.3 |            107.4 |            214.3 |
| SS claims only (panel labels) |         26.8 |             72.5 |             83.1 |
| A claims only                 |         38.3 |             76.5 |             90   |
| E OpenRouter tokens x price   |         -5   |            -23   |             33.3 |
| G GitHub new-repo share       |         -1.4 |             33.8 |             35.3 |
| naive last claim              |          3.8 |              0   |            -12.3 |
| P panel pca (pooled)          |         21.8 |             57.5 |             60.9 |
| P panel ridge (pooled)        |          4.2 |             26.3 |             29.1 |
| trend (pooled)                |         37.5 |             92.5 |            119.9 |
| ridge (pooled)                |          4.2 |             26.3 |             29.1 |
| lasso (pooled)                |         24.2 |             62.6 |             74.1 |
| enet (pooled)                 |         12.1 |             38.2 |             42   |
| pca (pooled)                  |         21.8 |             57.5 |             60.9 |
| pls (pooled)                  |          6.5 |             27.9 |             47   |
| bayes_ridge (pooled)          |         -2.4 |             14.6 |             13.5 |
| ard (pooled)                  |         -2   |             14.2 |              9.9 |
| kernel_ridge (pooled)         |         36.2 |             91.5 |            117.4 |
| gp (pooled)                   |         37.1 |             91.5 |            117.4 |
| knn (pooled)                  |         36.3 |             79.5 |             98.3 |
| rf (pooled)                   |         22.3 |             52.3 |             60.5 |
| extra_trees (pooled)          |         32   |             78.3 |            106.2 |
| hgb (pooled)                  |         20.8 |             56.7 |             57   |
| lightgbm (pooled)             |         17   |             44.6 |             43.3 |
| xgboost (pooled)              |         16.4 |             44.4 |             41.2 |
| mlp (pooled)                  |         21   |             52.9 |             84.7 |
| midas (pooled)                |         31.6 |             79.3 |             95.4 |

Paired differences in |log error| (`B vs A`: negative = B better; 90% bootstrap interval):

| rule        | comparison                                      |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:------------|:------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| publication | AUTO (pooled) vs A claims only                  |  14 |      -0.057 |    -0.13  |     0.014 |          8 |       6 |
| publication | AUTO (pooled) vs E OpenRouter tokens x price    |  14 |       0.046 |    -0.004 |     0.1   |          7 |       7 |
| publication | AUTO (pooled) vs G GitHub new-repo share        |  14 |       0.051 |     0.017 |     0.089 |          4 |      10 |
| publication | AUTO (pooled) vs naive last claim               |  14 |      -0.07  |    -0.132 |    -0.009 |         11 |       3 |
| publication | AUTO (pooled) vs SS claims only (panel labels)  |  14 |      -0.061 |    -0.122 |    -0.001 |          8 |       5 |
| publication | AUTO (pooled) vs P panel pca (pooled)           |  14 |      -0.04  |    -0.078 |    -0.007 |          4 |       2 |
| publication | AUTO (pooled) vs P panel ridge (pooled)         |  14 |       0.014 |    -0.005 |     0.035 |          2 |       3 |
| publication | stack (pooled) vs A claims only                 |  14 |      -0.032 |    -0.081 |     0.022 |          8 |       6 |
| publication | stack (pooled) vs E OpenRouter tokens x price   |  14 |       0.071 |     0.023 |     0.119 |          5 |       9 |
| publication | stack (pooled) vs G GitHub new-repo share       |  14 |       0.076 |     0.048 |     0.105 |          2 |      12 |
| publication | stack (pooled) vs naive last claim              |  14 |      -0.045 |    -0.132 |     0.042 |         11 |       3 |
| publication | stack (pooled) vs SS claims only (panel labels) |  14 |      -0.036 |    -0.078 |     0.006 |          8 |       5 |
| publication | AUTO (pooled) vs AUTO (own)                     |  14 |      -0.123 |    -0.248 |    -0.02  |          5 |       2 |
| publication | stack (pooled) vs stack (own)                   |  14 |      -0.099 |    -0.196 |    -0.021 |          5 |       2 |
| publication | AUTO (pooled) vs stack (pooled)                 |  14 |      -0.025 |    -0.054 |     0.002 |          4 |       3 |

Per-target log error:

| target         |   AUTO (pooled) |   stack (pooled) |   AUTO (own) |   stack (own) |   SS claims only (panel labels) |   A claims only |   E OpenRouter tokens x price |   G GitHub new-repo share |   naive last claim |
|:---------------|----------------:|-----------------:|-------------:|--------------:|--------------------------------:|----------------:|------------------------------:|--------------------------:|-------------------:|
| rev_2025fy     |           0.08  |            0.08  |        0.08  |         0.08  |                           0.073 |           0.062 |                         0.014 |                    -0.022 |              0.118 |
| rev_2025q1     |           0.098 |            0.098 |        0.098 |         0.098 |                           0.091 |           0.082 |                         0.166 |                    -0.023 |              0.145 |
| rev_2025q2     |           0.028 |            0.028 |        0.028 |         0.028 |                           0.02  |           0.067 |                         0.029 |                    -0.012 |              0.149 |
| rev_2025q3     |           0.041 |            0.041 |        0.041 |         0.041 |                           0.03  |           0.043 |                        -0.184 |                    -0.042 |              0.158 |
| rev_2026q1     |           0.044 |            0.044 |        0.044 |         0.044 |                           0.044 |           0.05  |                         0.048 |                    -0.05  |              0.138 |
| rev_2026q2     |          -0.01  |           -0.01  |       -0.01  |        -0.01  |                          -0.01  |          -0.004 |                        -0.008 |                     0.005 |              0.056 |
| rev_2026q3     |          -0.041 |           -0.161 |       -0.408 |        -0.408 |                          -0.237 |          -0.324 |                         0.051 |                     0.014 |             -0.038 |
| rr_2025_10     |         nan     |          nan     |      nan     |       nan     |                         nan     |          -0.44  |                        -0.517 |                    -0.113 |              0.847 |
| rr_2025_12     |          -0.171 |           -0.171 |       -0.171 |        -0.171 |                          -0.171 |          -0.163 |                        -0.003 |                    -0.039 |              0.251 |
| rr_2026_02     |           0.359 |            0.298 |        0.36  |         0.36  |                           0.374 |           0.363 |                         0.096 |                     0.113 |              0.693 |
| rr_2026_03     |           0.238 |            0.238 |        0.214 |         0.216 |                           0.242 |           0.217 |                         0.087 |                     0.18  |              0.305 |
| rr_2026_04     |           0.379 |            0.385 |        0.458 |         0.419 |                           0.207 |           0.108 |                         0.108 |                     0.197 |              0.457 |
| rr_2026_05     |           0.128 |            0.105 |       -0.039 |         0.091 |                          -0.31  |          -0.206 |                         0.12  |                     0.098 |              0.449 |
| rr_2026_09_s1a |          -0.233 |           -0.384 |       -0.729 |        -0.729 |                          -0.545 |          -0.568 |                         0.262 |                    -0.291 |              0     |
| rr_2026_10_s1a |          -0.256 |           -0.413 |       -1.145 |        -1.145 |                          -0.605 |          -0.642 |                        -0.288 |                    -0.302 |              0.132 |

### Vintage rule: period_end (g_spec)

All models, own targets and common targets (bias: mean log error, + = forecast too low):

| model                         |   n |   mae_log |   n_common |   mae_log_common |   bias_log_common |   mae_log_h2_2026 |
|:------------------------------|----:|----------:|-----------:|-----------------:|------------------:|------------------:|
| AUTO (pooled)                 |  11 |     0.224 |         11 |            0.224 |             0.004 |             0.326 |
| stack (pooled)                |  11 |     0.265 |         11 |            0.265 |            -0.063 |             0.508 |
| AUTO (own)                    |  11 |     0.337 |         11 |            0.337 |            -0.112 |             0.745 |
| stack (own)                   |  11 |     0.335 |         11 |            0.335 |            -0.106 |             0.745 |
| SS claims only (panel labels) |  11 |     0.238 |         11 |            0.238 |            -0.08  |             0.357 |
| A claims only                 |  13 |     0.248 |         11 |            0.224 |            -0.092 |             0.452 |
| E OpenRouter tokens x price   |  13 |     0.2   |         11 |            0.156 |             0.044 |             0.337 |
| G GitHub new-repo share       |  13 |     0.156 |         11 |            0.151 |             0.125 |             0.047 |
| naive last claim              |  13 |     0.329 |         11 |            0.257 |             0.25  |             0.056 |
| P panel pca (pooled)          |  11 |     0.312 |         11 |            0.312 |            -0.09  |             0.662 |
| P panel ridge (pooled)        |  11 |     0.221 |         11 |            0.221 |            -0.01  |             0.346 |

Every member, mean |log error| on the targets AUTO forecast. The best member is selected after
the fact and therefore optimistic; ranks are among the members only:

| model        |   pooled |   own |   rank_pooled |   rank_own |
|:-------------|---------:|------:|--------------:|-----------:|
| ard          |    0.185 | 0.217 |             1 |          1 |
| bayes_ridge  |    0.189 | 0.217 |             2 |          2 |
| lightgbm     |    0.195 | 0.242 |             3 |          5 |
| xgboost      |    0.2   | 0.235 |             4 |          3 |
| rf           |    0.203 | 0.254 |             5 |          7 |
| enet         |    0.206 | 0.265 |             6 |         12 |
| hgb          |    0.209 | 0.238 |             7 |          4 |
| ridge        |    0.221 | 0.26  |             8 |          8 |
| AUTO         |    0.224 | 0.337 |           nan |        nan |
| midas        |    0.231 | 0.26  |             9 |          9 |
| pls          |    0.234 | 0.262 |            10 |         10 |
| lasso        |    0.249 | 0.247 |            11 |          6 |
| knn          |    0.253 | 0.311 |            12 |         16 |
| mlp          |    0.255 | 0.314 |            13 |         18 |
| extra_trees  |    0.258 | 0.266 |            14 |         13 |
| stack        |    0.265 | 0.335 |           nan |        nan |
| gp           |    0.268 | 0.262 |            15 |         11 |
| kernel_ridge |    0.27  | 0.267 |            16 |         14 |
| trend        |    0.293 | 0.293 |            17 |         15 |
| pca          |    0.312 | 0.314 |            18 |         17 |

Spread of member scores, AUTO's rank, and a leave-one-target-out best-of-zoo (pick the member
with the lowest error on the other targets, score it on the held-out one):

| pool   |   n_targets |   n_members | best member (optimistic)   |   best |   p25 |   median |   p75 |   worst | worst member   |   AUTO |   AUTO rank among members |   stack |   LOO best-of-zoo |
|:-------|------------:|------------:|:---------------------------|-------:|------:|---------:|------:|--------:|:---------------|-------:|--------------------------:|--------:|------------------:|
| pooled |          11 |          18 | ard                        |  0.185 | 0.204 |    0.233 | 0.257 |   0.312 | pca            |  0.224 |                         9 |   0.265 |             0.245 |
| own    |          11 |          18 | ard                        |  0.217 | 0.244 |    0.261 | 0.267 |   0.314 | mlp            |  0.337 |                        19 |   0.335 |             0.292 |

AUTO's pick and log error per target:

| target         | vintage    |   gap_weeks | own   | pooled       |   log_err AUTO own |   log_err AUTO pooled |
|:---------------|:-----------|------------:|:------|:-------------|-------------------:|----------------------:|
| rr_2025_12     | 2025-12-28 |        9.29 | trend | trend        |              0.065 |                 0.065 |
| rev_2025fy     | 2025-12-31 |       11.71 | trend | trend        |             -0.017 |                -0.017 |
| rr_2026_02     | 2026-02-09 |       15.43 | trend | pca          |              0.36  |                 0.359 |
| rr_2026_03     | 2026-02-22 |        1.86 | mlp   | rf           |              0.214 |                 0.238 |
| rr_2026_04     | 2026-03-20 |        3.71 | pca   | rf           |              0.458 |                 0.379 |
| rev_2026q1     | 2026-03-31 |        7.29 | pca   | rf           |              0.139 |                 0.088 |
| rr_2026_05     | 2026-05-15 |        8    | mlp   | kernel_ridge |             -0.039 |                 0.128 |
| rev_2026q2     | 2026-06-30 |        8.57 | mlp   | ridge        |             -0.174 |                -0.215 |
| rev_2026q3     | 2026-09-30 |       11.14 | mlp   | extra_trees  |             -0.411 |                -0.267 |
| rr_2026_09_s1a | 2026-09-30 |        9.14 | mlp   | extra_trees  |             -0.703 |                -0.526 |
| rr_2026_10_s1a | 2026-10-31 |       13.57 | mlp   | ridge        |             -1.121 |                -0.184 |

H2 2026 targets, % error (+ = forecast too high):

| model                         |   rev_2026q3 |   rr_2026_09_s1a |   rr_2026_10_s1a |
|:------------------------------|-------------:|-----------------:|-----------------:|
| AUTO (pooled)                 |         30.5 |             69.2 |             20.2 |
| stack (pooled)                |         50.7 |            108.9 |             45.9 |
| AUTO (own)                    |         50.9 |            102.1 |            206.7 |
| stack (own)                   |         50.9 |            102.1 |            206.7 |
| SS claims only (panel labels) |         26.8 |             51.6 |             52   |
| A claims only                 |         38.3 |             64   |             71   |
| E OpenRouter tokens x price   |        -20.8 |            -31.3 |             49.4 |
| G GitHub new-repo share       |          1.2 |             13.2 |              0.5 |
| naive last claim              |          3.8 |              0   |            -12.3 |
| P panel pca (pooled)          |         78.6 |            161.9 |             55.9 |
| P panel ridge (pooled)        |         33.4 |             76.1 |             20.2 |
| trend (pooled)                |         37.5 |             86.5 |            113.2 |
| ridge (pooled)                |         33.4 |             76.1 |             20.2 |
| lasso (pooled)                |         24.3 |             57.9 |             68.5 |
| enet (pooled)                 |         14.8 |             38.8 |             37.2 |
| pca (pooled)                  |         78.6 |            161.9 |             55.9 |
| pls (pooled)                  |         33.2 |             71.3 |             37.4 |
| bayes_ridge (pooled)          |         13   |             35.6 |              5   |
| ard (pooled)                  |         11.3 |             32.9 |              5.8 |
| kernel_ridge (pooled)         |         37.4 |             86.1 |            111.4 |
| gp (pooled)                   |         37.4 |             86.1 |            111.4 |
| knn (pooled)                  |         34.2 |             79.8 |             92.8 |
| rf (pooled)                   |         21.3 |             50.1 |             59.8 |
| extra_trees (pooled)          |         30.5 |             69.2 |            100.6 |
| hgb (pooled)                  |         19.5 |             46.2 |             54.3 |
| lightgbm (pooled)             |         20.9 |             47.5 |             36.5 |
| xgboost (pooled)              |         21.1 |             50.4 |             37   |
| mlp (pooled)                  |         37.6 |             88.1 |             66.9 |
| midas (pooled)                |         35   |             81.4 |             88.8 |

Paired differences in |log error| (`B vs A`: negative = B better; 90% bootstrap interval):

| rule       | comparison                                      |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:-----------|:------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| period_end | AUTO (pooled) vs A claims only                  |  11 |       0     |    -0.073 |     0.068 |          5 |       6 |
| period_end | AUTO (pooled) vs E OpenRouter tokens x price    |  11 |       0.069 |     0.001 |     0.135 |          3 |       8 |
| period_end | AUTO (pooled) vs G GitHub new-repo share        |  11 |       0.073 |    -0.026 |     0.17  |          4 |       7 |
| period_end | AUTO (pooled) vs naive last claim               |  11 |      -0.032 |    -0.146 |     0.092 |          7 |       4 |
| period_end | AUTO (pooled) vs SS claims only (panel labels)  |  11 |      -0.013 |    -0.069 |     0.04  |          6 |       5 |
| period_end | AUTO (pooled) vs P panel pca (pooled)           |  11 |      -0.088 |    -0.172 |    -0.018 |          6 |       4 |
| period_end | AUTO (pooled) vs P panel ridge (pooled)         |  11 |       0.003 |    -0.015 |     0.028 |          7 |       2 |
| period_end | stack (pooled) vs A claims only                 |  11 |       0.041 |    -0.021 |     0.106 |          4 |       7 |
| period_end | stack (pooled) vs E OpenRouter tokens x price   |  11 |       0.11  |     0.048 |     0.177 |          3 |       8 |
| period_end | stack (pooled) vs G GitHub new-repo share       |  11 |       0.114 |    -0.011 |     0.245 |          4 |       7 |
| period_end | stack (pooled) vs naive last claim              |  11 |       0.009 |    -0.143 |     0.174 |          7 |       4 |
| period_end | stack (pooled) vs SS claims only (panel labels) |  11 |       0.028 |    -0.037 |     0.099 |          7 |       4 |
| period_end | AUTO (pooled) vs AUTO (own)                     |  11 |      -0.112 |    -0.268 |    -0.005 |          7 |       4 |
| period_end | stack (pooled) vs stack (own)                   |  11 |      -0.07  |    -0.2   |     0.009 |          6 |       5 |
| period_end | AUTO (pooled) vs stack (pooled)                 |  11 |      -0.041 |    -0.089 |     0.002 |          7 |       4 |

Per-target log error:

| target         |   AUTO (pooled) |   stack (pooled) |   AUTO (own) |   stack (own) |   SS claims only (panel labels) |   A claims only |   E OpenRouter tokens x price |   G GitHub new-repo share |   naive last claim |
|:---------------|----------------:|-----------------:|-------------:|--------------:|--------------------------------:|----------------:|------------------------------:|--------------------------:|-------------------:|
| rev_2025fy     |          -0.017 |           -0.038 |       -0.017 |        -0.017 |                          -0.05  |          -0.034 |                        -0.04  |                     0.127 |              0.145 |
| rev_2025q3     |         nan     |          nan     |      nan     |       nan     |                         nan     |          -0.315 |                        -0.378 |                    -0.252 |              0.603 |
| rev_2026q1     |           0.088 |            0.09  |        0.139 |         0.112 |                          -0.068 |          -0.021 |                        -0.04  |                     0.118 |              0.237 |
| rev_2026q2     |          -0.215 |           -0.225 |       -0.174 |        -0.174 |                          -0.247 |          -0.12  |                        -0.127 |                     0.558 |              0.115 |
| rev_2026q3     |          -0.267 |           -0.41  |       -0.411 |        -0.411 |                          -0.237 |          -0.324 |                         0.234 |                    -0.012 |             -0.038 |
| rr_2025_10     |         nan     |          nan     |      nan     |       nan     |                         nan     |          -0.44  |                        -0.517 |                    -0.113 |              0.847 |
| rr_2025_12     |           0.065 |           -0.016 |        0.065 |         0.065 |                           0.043 |           0.04  |                        -0.003 |                     0.101 |              0.251 |
| rr_2026_02     |           0.359 |            0.298 |        0.36  |         0.36  |                           0.374 |           0.363 |                         0.097 |                     0.113 |              0.693 |
| rr_2026_03     |           0.238 |            0.238 |        0.214 |         0.216 |                           0.242 |           0.217 |                         0.141 |                     0.211 |              0.305 |
| rr_2026_04     |           0.379 |            0.382 |        0.458 |         0.419 |                           0.207 |           0.108 |                         0.081 |                     0.192 |              0.457 |
| rr_2026_05     |           0.128 |            0.105 |       -0.039 |         0.091 |                          -0.31  |          -0.206 |                         0.171 |                     0.098 |              0.449 |
| rr_2026_09_s1a |          -0.526 |           -0.737 |       -0.703 |        -0.703 |                          -0.416 |          -0.494 |                         0.376 |                    -0.124 |              0     |
| rr_2026_10_s1a |          -0.184 |           -0.378 |       -1.121 |        -1.121 |                          -0.419 |          -0.537 |                        -0.401 |                    -0.005 |              0.132 |

### Stack weights (mean over targets; members with any mean weight > 0.005)

|                           |   trend |   ridge |   pca |   pls |   kernel_ridge |    gp |   knn |    rf |   hgb |   lightgbm |   mlp |   midas |
|:--------------------------|--------:|--------:|------:|------:|---------------:|------:|------:|------:|------:|-----------:|------:|--------:|
| ('period_end', 'own')     |   0.273 |   0     | 0     | 0.085 |          0.091 | 0     | 0     | 0     | 0     |      0     | 0.447 |   0.105 |
| ('period_end', 'pooled')  |   0.045 |   0.03  | 0.162 | 0     |          0.102 | 0.104 | 0.041 | 0.359 | 0.079 |      0.078 | 0     |   0     |
| ('publication', 'own')    |   0.071 |   0     | 0     | 0.033 |          0.071 | 0     | 0     | 0     | 0     |      0     | 0.774 |   0.05  |
| ('publication', 'pooled') |   0.036 |   0.212 | 0     | 0     |          0.06  | 0.023 | 0.197 | 0.17  | 0.071 |      0.233 | 0     |   0     |

### AUTO's picks (number of Anthropic targets)

|                           |   extra_trees |   kernel_ridge |   mlp |   pca |   rf |   ridge |   trend |
|:--------------------------|--------------:|---------------:|------:|------:|-----:|--------:|--------:|
| ('period_end', 'own')     |             0 |              0 |     6 |     2 |    0 |       0 |       3 |
| ('period_end', 'pooled')  |             2 |              1 |     0 |     1 |    3 |       2 |       2 |
| ('publication', 'own')    |             0 |              0 |    12 |     1 |    0 |       0 |       1 |
| ('publication', 'pooled') |             0 |              1 |     0 |     1 |    3 |       9 |       0 |

## Second test set: OpenAI claims (g_spec)

OpenAI's included, non-forecast point total claims (all tiers), pooled training only. The naive last
claim here is the panel's own (`naive last claim (panel)`, neutral crossing position).

### publication

| model                         |   n |   mae_log |   n_common |   mae_log_common |   bias_log_common |
|:------------------------------|----:|----------:|-----------:|-----------------:|------------------:|
| AUTO (pooled)                 |  15 |     0.106 |         15 |            0.106 |            -0.025 |
| stack (pooled)                |  15 |     0.097 |         15 |            0.097 |            -0.025 |
| SS claims only (panel labels) |  16 |     0.124 |         15 |            0.101 |            -0.008 |
| naive last claim (panel)      |  16 |     0.217 |         15 |            0.16  |             0.143 |

Members (pooled), spread and AUTO's rank:

| model        |   pooled |   rank_pooled |
|:-------------|---------:|--------------:|
| kernel_ridge |    0.079 |             1 |
| trend        |    0.08  |             2 |
| lasso        |    0.08  |             3 |
| knn          |    0.082 |             4 |
| ridge        |    0.083 |             5 |
| enet         |    0.083 |             6 |
| ard          |    0.086 |             7 |
| midas        |    0.09  |             8 |
| gp           |    0.092 |             9 |
| pca          |    0.093 |            10 |
| mlp          |    0.095 |            11 |
| pls          |    0.095 |            12 |
| bayes_ridge  |    0.095 |            13 |
| extra_trees  |    0.096 |            14 |
| stack        |    0.097 |           nan |
| rf           |    0.101 |            15 |
| AUTO         |    0.106 |           nan |
| xgboost      |    0.108 |            16 |
| lightgbm     |    0.109 |            17 |
| hgb          |    0.112 |            18 |

| pool   |   n_targets |   n_members | best member (optimistic)   |   best |   p25 |   median |   p75 |   worst | worst member   |   AUTO |   AUTO rank among members |   stack |   LOO best-of-zoo |
|:-------|------------:|------------:|:---------------------------|-------:|------:|---------:|------:|--------:|:---------------|-------:|--------------------------:|--------:|------------------:|
| pooled |          15 |          18 | kernel_ridge               |  0.079 | 0.083 |    0.092 | 0.095 |   0.112 | hgb            |  0.106 |                        16 |   0.097 |             0.108 |

| rule        | comparison                                      |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:------------|:------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| publication | AUTO (pooled) vs SS claims only (panel labels)  |  15 |       0.005 |    -0.011 |     0.022 |          7 |       7 |
| publication | AUTO (pooled) vs naive last claim (panel)       |  15 |      -0.054 |    -0.109 |     0.001 |         10 |       5 |
| publication | stack (pooled) vs SS claims only (panel labels) |  15 |      -0.004 |    -0.019 |     0.011 |          8 |       6 |
| publication | stack (pooled) vs naive last claim (panel)      |  15 |      -0.063 |    -0.12  |    -0.007 |         10 |       5 |
| publication | AUTO (pooled) vs stack (pooled)                 |  15 |       0.009 |     0     |     0.021 |          2 |       8 |

### period_end

| model                         |   n |   mae_log |   n_common |   mae_log_common |   bias_log_common |
|:------------------------------|----:|----------:|-----------:|-----------------:|------------------:|
| AUTO (pooled)                 |  15 |     0.085 |         15 |            0.085 |             0.001 |
| stack (pooled)                |  15 |     0.079 |         15 |            0.079 |            -0.007 |
| SS claims only (panel labels) |  16 |     0.1   |         15 |            0.075 |             0.031 |
| naive last claim (panel)      |  16 |     0.254 |         15 |            0.199 |             0.199 |

Members (pooled), spread and AUTO's rank:

| model        |   pooled |   rank_pooled |
|:-------------|---------:|--------------:|
| trend        |    0.073 |             1 |
| knn          |    0.074 |             2 |
| lasso        |    0.074 |             3 |
| enet         |    0.075 |             4 |
| midas        |    0.076 |             5 |
| ard          |    0.076 |             6 |
| ridge        |    0.078 |             7 |
| kernel_ridge |    0.079 |             8 |
| stack        |    0.079 |           nan |
| gp           |    0.083 |             9 |
| extra_trees  |    0.084 |            10 |
| AUTO         |    0.085 |           nan |
| mlp          |    0.086 |            11 |
| rf           |    0.086 |            12 |
| bayes_ridge  |    0.086 |            13 |
| pca          |    0.089 |            14 |
| xgboost      |    0.091 |            15 |
| hgb          |    0.091 |            16 |
| pls          |    0.091 |            17 |
| lightgbm     |    0.095 |            18 |

| pool   |   n_targets |   n_members | best member (optimistic)   |   best |   p25 |   median |   p75 |   worst | worst member   |   AUTO |   AUTO rank among members |   stack |   LOO best-of-zoo |
|:-------|------------:|------------:|:---------------------------|-------:|------:|---------:|------:|--------:|:---------------|-------:|--------------------------:|--------:|------------------:|
| pooled |          15 |          18 | trend                      |  0.073 | 0.076 |    0.084 | 0.089 |   0.095 | lightgbm       |  0.085 |                        11 |   0.079 |             0.106 |

| rule       | comparison                                      |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:-----------|:------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| period_end | AUTO (pooled) vs SS claims only (panel labels)  |  15 |       0.01  |    -0.009 |     0.029 |          5 |      10 |
| period_end | AUTO (pooled) vs naive last claim (panel)       |  15 |      -0.113 |    -0.194 |    -0.037 |         12 |       3 |
| period_end | stack (pooled) vs SS claims only (panel labels) |  15 |       0.004 |    -0.015 |     0.023 |          6 |       9 |
| period_end | stack (pooled) vs naive last claim (panel)      |  15 |      -0.119 |    -0.197 |    -0.045 |         11 |       4 |
| period_end | AUTO (pooled) vs stack (pooled)                 |  15 |       0.006 |    -0.003 |     0.018 |          8 |       7 |

AUTO's picks (number of OpenAI targets):

|                           |   extra_trees |   gp |   hgb |   kernel_ridge |   lightgbm |   rf |   ridge |   trend |   xgboost |
|:--------------------------|--------------:|-----:|------:|---------------:|-----------:|-----:|--------:|--------:|----------:|
| ('period_end', 'pooled')  |             2 |    0 |     1 |              0 |          1 |    4 |       1 |       6 |         0 |
| ('publication', 'pooled') |             3 |    1 |     2 |              1 |          1 |    4 |       1 |       1 |         1 |

## Run notes

- Packages: scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1. Members dropped: none.
- Every member single-threaded (OMP_NUM_THREADS=1); 4 worker processes across targets.
- g_spec / publication: 30 targets forecast; summed job time 14.7 CPU-minutes (about a quarter of that in wall time on 4 workers).
- g_spec / period_end: 27 targets forecast; summed job time 12.7 CPU-minutes (about a quarter of that in wall time on 4 workers).
- The MIDAS member was revised once before the first backtest (see the protocol in `model_zoo.py`): the
  first draft put lags 0..13 under a single Almon polynomial, which cannot represent a contemporaneous-only
  effect; a synthetic unit test showed this. No test target had been forecast at that point.
- The naive last claim for Anthropic comes from `forecast_backtest_*` (crossing claims re-dated per vintage);
  the panel's own naive (neutral crossing position) is in the CSVs as `naive last claim (panel)`.
- Before t0 (the last anchor public at the vintage) every member returns the label path, so members can
  differ only on targets past t0; with few such targets, differences between members are noisy.

