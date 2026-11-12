# Pooled two-company panel model: backtest (2026-11-10 data)

Specification: the docstring of `panel_model.py`, fixed before the first run. Labels: the claims-only
state-space path of each company, refit on claims public at the vintage; features: changes in
`features.asof_features(v)`; targets: weekly log growth over 4, 8 and 13 weeks; forecast: label at the
last anchor public at v plus predicted growth. Filing rows are never inputs. Error = log(actual / forecast),
negative = forecast too high. `pooled` = Anthropic + OpenAI training pairs, `own` = target company only
(the pre-declared control). Existing variants are read through `forecast_eval.collect()`; the naive last
claim is `backtest._naive` from `forecast_backtest_<rule>_2026-11-10.csv`.

## Anthropic targets: mean |log error| by rule

`mae_log`: each model's own targets; `mae_log_common`: targets every listed model forecast;
`mae_log_h2_2026`: rev_2026q3, rr_2026_09_s1a, rr_2026_10_s1a.

| model                         |   publication: n |   publication: mae_log |   publication: mae_log_common |   publication: mae_log_h2_2026 |   period_end: n |   period_end: mae_log |   period_end: mae_log_common |   period_end: mae_log_h2_2026 |
|:------------------------------|-----------------:|-----------------------:|------------------------------:|-------------------------------:|----------------:|----------------------:|-----------------------------:|------------------------------:|
| panel trend (pooled)          |               14 |                  0.239 |                         0.239 |                          0.587 |              11 |                 0.293 |                        0.293 |                         0.566 |
| panel pca (pooled)            |               14 |                  0.19  |                         0.19  |                          0.376 |              11 |                 0.312 |                        0.312 |                         0.662 |
| panel ridge (pooled)          |               14 |                  0.137 |                         0.137 |                          0.177 |              11 |                 0.221 |                        0.221 |                         0.346 |
| panel gbm (pooled)            |               14 |                  0.175 |                         0.175 |                          0.363 |              11 |                 0.209 |                        0.209 |                         0.331 |
| panel trend (own)             |               14 |                  0.24  |                         0.24  |                          0.584 |              11 |                 0.293 |                        0.293 |                         0.564 |
| panel pca (own)               |               14 |                  0.247 |                         0.247 |                          0.665 |              11 |                 0.314 |                        0.314 |                         0.694 |
| panel ridge (own)             |               14 |                  0.215 |                         0.215 |                          0.524 |              11 |                 0.26  |                        0.26  |                         0.502 |
| panel gbm (own)               |               14 |                  0.192 |                         0.192 |                          0.467 |              11 |                 0.238 |                        0.238 |                         0.443 |
| SS claims only (panel labels) |               14 |                  0.211 |                         0.211 |                          0.462 |              11 |                 0.238 |                        0.238 |                         0.357 |
| A claims only                 |               15 |                  0.223 |                         0.207 |                          0.512 |              13 |                 0.248 |                        0.224 |                         0.452 |
| E OpenRouter tokens x price   |               15 |                  0.132 |                         0.105 |                          0.2   |              13 |                 0.2   |                        0.156 |                         0.337 |
| G GitHub new-repo share       |               15 |                  0.1   |                         0.099 |                          0.203 |              13 |                 0.156 |                        0.151 |                         0.047 |
| ML all signals (smoother)     |               14 |                  0.19  |                         0.19  |                          0.452 |              11 |                 0.243 |                        0.243 |                         0.429 |
| naive last claim              |               15 |                  0.262 |                         0.221 |                          0.056 |              13 |                 0.329 |                        0.257 |                         0.056 |

## Vintage rule: publication

| model                         |   n |   mae_log |   bias_log |   n_common |   mae_log_common |   mae_log_h2_2026 |
|:------------------------------|----:|----------:|-----------:|-----------:|-----------------:|------------------:|
| panel trend (pooled)          |  14 |     0.239 |     -0.038 |         14 |            0.239 |             0.587 |
| panel pca (pooled)            |  14 |     0.19  |      0.004 |         14 |            0.19  |             0.376 |
| panel ridge (pooled)          |  14 |     0.137 |      0.035 |         14 |            0.137 |             0.177 |
| panel gbm (pooled)            |  14 |     0.175 |     -0.034 |         14 |            0.175 |             0.363 |
| panel trend (own)             |  14 |     0.24  |     -0.036 |         14 |            0.24  |             0.584 |
| panel pca (own)               |  14 |     0.247 |     -0.064 |         14 |            0.247 |             0.665 |
| panel ridge (own)             |  14 |     0.215 |     -0.035 |         14 |            0.215 |             0.524 |
| panel gbm (own)               |  14 |     0.192 |     -0.062 |         14 |            0.192 |             0.467 |
| SS claims only (panel labels) |  14 |     0.211 |     -0.057 |         14 |            0.211 |             0.462 |
| A claims only                 |  15 |     0.223 |     -0.09  |         14 |            0.207 |             0.512 |
| E OpenRouter tokens x price   |  15 |     0.132 |     -0.001 |         14 |            0.105 |             0.2   |
| G GitHub new-repo share       |  15 |     0.1   |     -0.019 |         14 |            0.099 |             0.203 |
| ML all signals (smoother)     |  14 |     0.19  |     -0.129 |         14 |            0.19  |             0.452 |
| naive last claim              |  15 |     0.262 |      0.257 |         14 |            0.221 |             0.056 |

H2 2026 targets, % error (+ = forecast too high):

| model                         |   rev_2026q3 |   rr_2026_09_s1a |   rr_2026_10_s1a |
|:------------------------------|-------------:|-----------------:|-----------------:|
| panel trend (pooled)          |         37.5 |             92.5 |            119.9 |
| panel pca (pooled)            |         21.8 |             57.5 |             60.9 |
| panel ridge (pooled)          |          4.2 |             26.3 |             29.1 |
| panel gbm (pooled)            |         20.8 |             56.7 |             57   |
| panel trend (own)             |         37.5 |             91.9 |            118.6 |
| panel pca (own)               |         49.6 |            104.5 |            140.1 |
| panel ridge (own)             |         33.6 |             80.7 |             99.4 |
| panel gbm (own)               |         31.4 |             68   |             83.8 |
| SS claims only (panel labels) |         26.8 |             72.5 |             83.1 |
| A claims only                 |         38.3 |             76.5 |             90   |
| E OpenRouter tokens x price   |         -5   |            -23   |             33.3 |
| G GitHub new-repo share       |         -1.4 |             33.8 |             35.3 |
| ML all signals (smoother)     |         34.6 |             61.1 |             79.1 |
| naive last claim              |          3.8 |              0   |            -12.3 |

Per-target log error:

| target         |   panel trend (pooled) |   panel pca (pooled) |   panel ridge (pooled) |   panel gbm (pooled) |   panel pca (own) |   SS claims only (panel labels) |   A claims only |   E OpenRouter tokens x price |   G GitHub new-repo share |   ML all signals (smoother) |   naive last claim |
|:---------------|-----------------------:|---------------------:|-----------------------:|---------------------:|------------------:|--------------------------------:|----------------:|------------------------------:|--------------------------:|----------------------------:|-------------------:|
| rev_2025fy     |                  0.08  |                0.08  |                  0.08  |                0.08  |             0.08  |                           0.073 |           0.062 |                         0.014 |                    -0.022 |                      -0.041 |              0.118 |
| rev_2025q1     |                  0.098 |                0.098 |                  0.098 |                0.098 |             0.098 |                           0.091 |           0.082 |                         0.166 |                    -0.023 |                      -0.018 |              0.145 |
| rev_2025q2     |                  0.028 |                0.028 |                  0.028 |                0.028 |             0.028 |                           0.02  |           0.067 |                         0.029 |                    -0.012 |                      -0.103 |              0.149 |
| rev_2025q3     |                  0.041 |                0.041 |                  0.041 |                0.041 |             0.041 |                           0.03  |           0.043 |                        -0.184 |                    -0.042 |                      -0.1   |              0.158 |
| rev_2026q1     |                  0.044 |                0.044 |                  0.044 |                0.044 |             0.044 |                           0.044 |           0.05  |                         0.048 |                    -0.05  |                      -0.081 |              0.138 |
| rev_2026q2     |                 -0.01  |               -0.01  |                 -0.01  |               -0.01  |            -0.01  |                          -0.01  |          -0.004 |                        -0.008 |                     0.005 |                      -0.121 |              0.056 |
| rev_2026q3     |                 -0.319 |               -0.197 |                 -0.041 |               -0.189 |            -0.403 |                          -0.237 |          -0.324 |                         0.051 |                     0.014 |                      -0.297 |             -0.038 |
| rr_2025_10     |                nan     |              nan     |                nan     |              nan     |           nan     |                         nan     |          -0.44  |                        -0.517 |                    -0.113 |                     nan     |              0.847 |
| rr_2025_12     |                 -0.171 |               -0.171 |                 -0.171 |               -0.171 |            -0.171 |                          -0.171 |          -0.163 |                        -0.003 |                    -0.039 |                      -0.149 |              0.251 |
| rr_2026_02     |                  0.365 |                0.359 |                  0.382 |                0.169 |             0.326 |                           0.374 |           0.363 |                         0.096 |                     0.113 |                      -0.108 |              0.693 |
| rr_2026_03     |                  0.24  |                0.207 |                  0.103 |                0.239 |             0.184 |                           0.242 |           0.217 |                         0.087 |                     0.18  |                      -0.161 |              0.305 |
| rr_2026_04     |                  0.389 |                0.397 |                  0.415 |                0.288 |             0.458 |                           0.207 |           0.108 |                         0.108 |                     0.197 |                       0.361 |              0.457 |
| rr_2026_05     |                  0.12  |                0.106 |                  0.015 |               -0.191 |             0.02  |                          -0.31  |          -0.206 |                         0.12  |                     0.098 |                       0.067 |              0.449 |
| rr_2026_09_s1a |                 -0.655 |               -0.454 |                 -0.233 |               -0.449 |            -0.715 |                          -0.545 |          -0.568 |                         0.262 |                    -0.291 |                      -0.477 |              0     |
| rr_2026_10_s1a |                 -0.788 |               -0.476 |                 -0.256 |               -0.451 |            -0.876 |                          -0.605 |          -0.642 |                        -0.288 |                    -0.302 |                      -0.583 |              0.132 |

Paired differences in |log error| (negative = second model better; 90% bootstrap interval):

| rule        | comparison                                          |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:------------|:----------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| publication | panel trend (pooled) vs panel trend (own)           |  14 |      -0.001 |    -0.005 |     0.001 |          2 |       5 |
| publication | panel pca (pooled) vs panel pca (own)               |  14 |      -0.056 |    -0.117 |    -0.003 |          4 |       3 |
| publication | panel ridge (pooled) vs panel ridge (own)           |  14 |      -0.078 |    -0.146 |    -0.018 |          5 |       2 |
| publication | panel gbm (pooled) vs panel gbm (own)               |  14 |      -0.018 |    -0.041 |     0.003 |          4 |       3 |
| publication | panel pca (pooled) vs panel trend (pooled)          |  14 |      -0.049 |    -0.092 |    -0.011 |          6 |       1 |
| publication | panel ridge (pooled) vs panel trend (pooled)        |  14 |      -0.102 |    -0.182 |    -0.032 |          5 |       2 |
| publication | panel gbm (pooled) vs panel trend (pooled)          |  14 |      -0.064 |    -0.113 |    -0.019 |          6 |       1 |
| publication | panel pca (own) vs panel trend (own)                |  14 |       0.006 |    -0.019 |     0.03  |          3 |       4 |
| publication | panel ridge (own) vs panel trend (own)              |  14 |      -0.025 |    -0.048 |    -0.003 |          6 |       1 |
| publication | panel gbm (own) vs panel trend (own)                |  14 |      -0.048 |    -0.086 |    -0.013 |          6 |       1 |
| publication | panel pca (pooled) vs A claims only                 |  14 |      -0.017 |    -0.058 |     0.031 |          9 |       5 |
| publication | panel pca (pooled) vs E OpenRouter tokens x price   |  14 |       0.086 |     0.032 |     0.14  |          5 |       9 |
| publication | panel pca (pooled) vs G GitHub new-repo share       |  14 |       0.091 |     0.055 |     0.128 |          2 |      12 |
| publication | panel pca (pooled) vs naive last claim              |  14 |      -0.03  |    -0.121 |     0.065 |         11 |       3 |
| publication | panel pca (pooled) vs ML all signals (smoother)     |  14 |       0     |    -0.039 |     0.044 |          7 |       7 |
| publication | panel pca (pooled) vs SS claims only (panel labels) |  14 |      -0.021 |    -0.057 |     0.016 |          8 |       5 |

## Vintage rule: period_end

| model                         |   n |   mae_log |   bias_log |   n_common |   mae_log_common |   mae_log_h2_2026 |
|:------------------------------|----:|----------:|-----------:|-----------:|-----------------:|------------------:|
| panel trend (pooled)          |  11 |     0.293 |     -0.06  |         11 |            0.293 |             0.566 |
| panel pca (pooled)            |  11 |     0.312 |     -0.09  |         11 |            0.312 |             0.662 |
| panel ridge (pooled)          |  11 |     0.221 |     -0.01  |         11 |            0.221 |             0.346 |
| panel gbm (pooled)            |  11 |     0.209 |     -0.071 |         11 |            0.209 |             0.331 |
| panel trend (own)             |  11 |     0.293 |     -0.056 |         11 |            0.293 |             0.564 |
| panel pca (own)               |  11 |     0.314 |     -0.1   |         11 |            0.314 |             0.694 |
| panel ridge (own)             |  11 |     0.26  |     -0.052 |         11 |            0.26  |             0.502 |
| panel gbm (own)               |  11 |     0.238 |     -0.121 |         11 |            0.238 |             0.443 |
| SS claims only (panel labels) |  11 |     0.238 |     -0.08  |         11 |            0.238 |             0.357 |
| A claims only                 |  13 |     0.248 |     -0.136 |         11 |            0.224 |             0.452 |
| E OpenRouter tokens x price   |  13 |     0.2   |     -0.031 |         11 |            0.156 |             0.337 |
| G GitHub new-repo share       |  13 |     0.156 |      0.078 |         11 |            0.151 |             0.047 |
| ML all signals (smoother)     |  11 |     0.243 |     -0.159 |         11 |            0.243 |             0.429 |
| naive last claim              |  13 |     0.329 |      0.323 |         11 |            0.257 |             0.056 |

H2 2026 targets, % error (+ = forecast too high):

| model                         |   rev_2026q3 |   rr_2026_09_s1a |   rr_2026_10_s1a |
|:------------------------------|-------------:|-----------------:|-----------------:|
| panel trend (pooled)          |         37.5 |             86.5 |            113.2 |
| panel pca (pooled)            |         78.6 |            161.9 |             55.9 |
| panel ridge (pooled)          |         33.4 |             76.1 |             20.2 |
| panel gbm (pooled)            |         19.5 |             46.2 |             54.3 |
| panel trend (own)             |         37.5 |             86.3 |            112.2 |
| panel pca (own)               |         49.6 |            108.9 |            156.3 |
| panel ridge (own)             |         33.5 |             75.7 |             92.5 |
| panel gbm (own)               |         30.6 |             63.4 |             77   |
| SS claims only (panel labels) |         26.8 |             51.6 |             52   |
| A claims only                 |         38.3 |             64   |             71   |
| E OpenRouter tokens x price   |        -20.8 |            -31.3 |             49.4 |
| G GitHub new-repo share       |          1.2 |             13.2 |              0.5 |
| ML all signals (smoother)     |         34.6 |             57.2 |             71.1 |
| naive last claim              |          3.8 |              0   |            -12.3 |

Per-target log error:

| target         |   panel trend (pooled) |   panel pca (pooled) |   panel ridge (pooled) |   panel gbm (pooled) |   panel pca (own) |   SS claims only (panel labels) |   A claims only |   E OpenRouter tokens x price |   G GitHub new-repo share |   ML all signals (smoother) |   naive last claim |
|:---------------|-----------------------:|---------------------:|-----------------------:|---------------------:|------------------:|--------------------------------:|----------------:|------------------------------:|--------------------------:|----------------------------:|-------------------:|
| rev_2025fy     |                 -0.017 |               -0.019 |                 -0.02  |               -0.049 |            -0.022 |                          -0.05  |          -0.034 |                        -0.04  |                     0.127 |                      -0.103 |              0.145 |
| rev_2025q3     |                nan     |              nan     |                nan     |              nan     |           nan     |                         nan     |          -0.315 |                        -0.378 |                    -0.252 |                     nan     |              0.603 |
| rev_2026q1     |                  0.092 |                0.095 |                  0.092 |                0.029 |             0.139 |                          -0.068 |          -0.021 |                        -0.04  |                     0.118 |                       0.039 |              0.237 |
| rev_2026q2     |                 -0.221 |               -0.204 |                 -0.215 |               -0.239 |            -0.176 |                          -0.247 |          -0.12  |                        -0.127 |                     0.558 |                      -0.328 |              0.115 |
| rev_2026q3     |                 -0.319 |               -0.58  |                 -0.289 |               -0.178 |            -0.403 |                          -0.237 |          -0.324 |                         0.234 |                    -0.012 |                      -0.297 |             -0.038 |
| rr_2025_10     |                nan     |              nan     |                nan     |              nan     |           nan     |                         nan     |          -0.44  |                        -0.517 |                    -0.113 |                     nan     |              0.847 |
| rr_2025_12     |                  0.065 |                0.059 |                  0.055 |               -0.057 |             0.047 |                           0.043 |           0.04  |                        -0.003 |                     0.101 |                      -0.226 |              0.251 |
| rr_2026_02     |                  0.365 |                0.359 |                  0.382 |                0.169 |             0.326 |                           0.374 |           0.363 |                         0.097 |                     0.113 |                      -0.108 |              0.693 |
| rr_2026_03     |                  0.24  |                0.207 |                  0.103 |                0.239 |             0.184 |                           0.242 |           0.217 |                         0.141 |                     0.211 |                      -0.161 |              0.305 |
| rr_2026_04     |                  0.396 |                0.399 |                  0.4   |                0.322 |             0.458 |                           0.207 |           0.108 |                         0.081 |                     0.192 |                       0.361 |              0.457 |
| rr_2026_05     |                  0.122 |                0.105 |                  0.13  |               -0.205 |             0.02  |                          -0.31  |          -0.206 |                         0.171 |                     0.098 |                       0.067 |              0.449 |
| rr_2026_09_s1a |                 -0.624 |               -0.963 |                 -0.566 |               -0.38  |            -0.737 |                          -0.416 |          -0.494 |                         0.376 |                    -0.124 |                      -0.452 |              0     |
| rr_2026_10_s1a |                 -0.757 |               -0.444 |                 -0.184 |               -0.434 |            -0.941 |                          -0.419 |          -0.537 |                        -0.401 |                    -0.005 |                      -0.537 |              0.132 |

Paired differences in |log error| (negative = second model better; 90% bootstrap interval):

| rule       | comparison                                          |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:-----------|:----------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| period_end | panel trend (pooled) vs panel trend (own)           |  11 |      -0     |    -0.005 |     0.004 |          3 |       8 |
| period_end | panel pca (pooled) vs panel pca (own)               |  11 |      -0.002 |    -0.095 |     0.077 |          4 |       7 |
| period_end | panel ridge (pooled) vs panel ridge (own)           |  11 |      -0.038 |    -0.12  |     0.023 |          6 |       5 |
| period_end | panel gbm (pooled) vs panel gbm (own)               |  11 |      -0.029 |    -0.06  |     0     |          6 |       5 |
| period_end | panel pca (pooled) vs panel trend (pooled)          |  11 |       0.02  |    -0.061 |     0.103 |          6 |       5 |
| period_end | panel ridge (pooled) vs panel trend (pooled)        |  11 |      -0.071 |    -0.166 |    -0.006 |          6 |       5 |
| period_end | panel gbm (pooled) vs panel trend (pooled)          |  11 |      -0.083 |    -0.145 |    -0.025 |          8 |       3 |
| period_end | panel pca (own) vs panel trend (own)                |  11 |       0.021 |    -0.021 |     0.064 |          5 |       6 |
| period_end | panel ridge (own) vs panel trend (own)              |  11 |      -0.033 |    -0.062 |    -0.003 |          8 |       3 |
| period_end | panel gbm (own) vs panel trend (own)                |  11 |      -0.054 |    -0.106 |    -0.005 |          7 |       4 |
| period_end | panel pca (pooled) vs A claims only                 |  11 |       0.088 |     0.008 |     0.177 |          5 |       6 |
| period_end | panel pca (pooled) vs E OpenRouter tokens x price   |  11 |       0.157 |     0.07  |     0.257 |          2 |       9 |
| period_end | panel pca (pooled) vs G GitHub new-repo share       |  11 |       0.161 |     0.005 |     0.33  |          5 |       6 |
| period_end | panel pca (pooled) vs naive last claim              |  11 |       0.056 |    -0.123 |     0.256 |          7 |       4 |
| period_end | panel pca (pooled) vs ML all signals (smoother)     |  11 |       0.069 |    -0.022 |     0.172 |          4 |       7 |
| period_end | panel pca (pooled) vs SS claims only (panel labels) |  11 |       0.075 |    -0.017 |     0.18  |          5 |       6 |

## Secondary test set: OpenAI claims

OpenAI's included, non-forecast point claims (all tiers), each forecast from data public at its vintage
with its own origin excluded. OpenAI's basis (4 weeks x 12, net of cloud partners) is never mixed with
Anthropic's: each company keeps its own level, only growth is pooled.

### publication

| model                         |   n |   mae_log |   bias_log |   n_common |   mae_log_common |
|:------------------------------|----:|----------:|-----------:|-----------:|-----------------:|
| panel trend (pooled)          |  15 |     0.08  |     -0.024 |         15 |            0.08  |
| panel pca (pooled)            |  15 |     0.093 |     -0.011 |         15 |            0.093 |
| panel ridge (pooled)          |  15 |     0.083 |     -0.022 |         15 |            0.083 |
| panel gbm (pooled)            |  15 |     0.112 |     -0.025 |         15 |            0.112 |
| panel trend (own)             |  15 |     0.105 |     -0.018 |         15 |            0.105 |
| panel pca (own)               |  15 |     0.115 |     -0.005 |         15 |            0.115 |
| panel ridge (own)             |  15 |     0.112 |     -0.001 |         15 |            0.112 |
| panel gbm (own)               |  15 |     0.099 |     -0.01  |         15 |            0.099 |
| SS claims only (panel labels) |  16 |     0.124 |      0.022 |         15 |            0.101 |
| naive last claim              |  16 |     0.217 |      0.201 |         15 |            0.16  |

| rule        | comparison                                          |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:------------|:----------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| publication | panel trend (pooled) vs panel trend (own)           |  15 |      -0.025 |    -0.053 |    -0.001 |          5 |       4 |
| publication | panel pca (pooled) vs panel pca (own)               |  15 |      -0.022 |    -0.052 |     0     |          5 |       4 |
| publication | panel ridge (pooled) vs panel ridge (own)           |  15 |      -0.029 |    -0.079 |     0.002 |          4 |       5 |
| publication | panel gbm (pooled) vs panel gbm (own)               |  15 |       0.012 |    -0.005 |     0.034 |          5 |       4 |
| publication | panel pca (pooled) vs SS claims only (panel labels) |  15 |      -0.008 |    -0.05  |     0.032 |          8 |       6 |
| publication | panel pca (pooled) vs naive last claim              |  15 |      -0.067 |    -0.138 |    -0.002 |          9 |       6 |

### period_end

| model                         |   n |   mae_log |   bias_log |   n_common |   mae_log_common |
|:------------------------------|----:|----------:|-----------:|-----------:|-----------------:|
| panel trend (pooled)          |  15 |     0.073 |     -0.016 |         15 |            0.073 |
| panel pca (pooled)            |  15 |     0.089 |      0.008 |         15 |            0.089 |
| panel ridge (pooled)          |  15 |     0.078 |     -0.013 |         15 |            0.078 |
| panel gbm (pooled)            |  15 |     0.091 |      0.003 |         15 |            0.091 |
| panel trend (own)             |  15 |     0.086 |      0.004 |         15 |            0.086 |
| panel pca (own)               |  15 |     0.109 |      0.022 |         15 |            0.109 |
| panel ridge (own)             |  15 |     0.104 |      0.043 |         15 |            0.104 |
| panel gbm (own)               |  15 |     0.08  |      0.018 |         15 |            0.08  |
| SS claims only (panel labels) |  16 |     0.1   |      0.059 |         15 |            0.075 |
| naive last claim              |  16 |     0.254 |      0.254 |         15 |            0.199 |

| rule       | comparison                                          |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:-----------|:----------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| period_end | panel trend (pooled) vs panel trend (own)           |  15 |      -0.013 |    -0.041 |     0.006 |          7 |       5 |
| period_end | panel pca (pooled) vs panel pca (own)               |  15 |      -0.02  |    -0.052 |     0.005 |          4 |       8 |
| period_end | panel ridge (pooled) vs panel ridge (own)           |  15 |      -0.026 |    -0.077 |     0.007 |          4 |       8 |
| period_end | panel gbm (pooled) vs panel gbm (own)               |  15 |       0.011 |    -0.002 |     0.032 |          5 |       7 |
| period_end | panel pca (pooled) vs SS claims only (panel labels) |  15 |       0.014 |    -0.034 |     0.056 |          4 |      11 |
| period_end | panel pca (pooled) vs naive last claim              |  15 |      -0.109 |    -0.197 |    -0.027 |         10 |       5 |

## Run notes

- Bugs fixed while running (no change to the specification):
  1. GBM crashed when a feature column had no value in a CV training fold (sklearn's binning rejects an
     all-missing column). Found by the unit tests before any backtest; such columns are now dropped inside
     `GBMPanel.fit` (they cannot split, so the model is the same).
  2. The first full run crashed in the period_end rule at vintages with no anchor at all (the empty anchor
     series had a non-date index). Fixed in `anchors`; the publication results, computed before the crash,
     are unchanged, and no period_end result existed before the fix.
  3. Runtime only: the four worker processes oversubscribed BLAS/OpenMP threads; run with
     OMP_NUM_THREADS=1 (about 3 minutes for both rules).
- Before t0 (the last anchor public at the vintage) every panel model returns the label path, so
  pooled and own forecasts can differ only on targets past t0 (about half the Anthropic set).
- The naive last claim here comes from `forecast_backtest_*`; for revenue targets it differs by up to 10%
  from the one the panel computes itself (`forecast_panel_*.csv`), because that backtest re-dates crossing
  claims per vintage (`crossing='profile'`), while the panel uses the neutral position 0.5.

## What the linear models learned (last Anthropic vintage, 2026-11-09)

Coefficients in original units: weekly log growth per unit of weekly feature change (log or share
points); `mom` = weight on the previous weeks' label growth. pca: coefficients implied by the kept components.

| vintage    | pool   | model   | hyper                     |   n_pairs |   mom |   openrouter_log_tokens |   openrouter_share |   github_repo_share |   ramp_adoption |   vercel_spend_share |   log_price |   npm_level |   pypi_level |
|:-----------|:-------|:--------|:--------------------------|----------:|------:|------------------------:|-------------------:|--------------------:|----------------:|---------------------:|------------:|------------:|-------------:|
| 2026-11-09 | pooled | trend   | {'alpha': 1000.0}         |       435 | 0.086 |                 nan     |            nan     |             nan     |         nan     |              nan     |     nan     |     nan     |      nan     |
| 2026-11-09 | pooled | pca     | {'alpha': 1000.0, 'k': 1} |       435 | 0.074 |                   0.059 |              0.224 |               0.102 |           0.176 |                0.842 |      -0.007 |       0.025 |        0.078 |
| 2026-11-09 | pooled | ridge   | {'alpha': 100.0}          |       435 | 0.132 |                  -0.027 |              0.332 |               1.114 |           2.085 |               -0.021 |       1.14  |       0.078 |       -0.103 |
| 2026-11-09 | own    | trend   | {'alpha': 1000.0}         |       190 | 0.038 |                 nan     |            nan     |             nan     |         nan     |              nan     |     nan     |     nan     |      nan     |
| 2026-11-09 | own    | pca     | {'alpha': 1000.0, 'k': 1} |       190 | 0.025 |                   0.004 |             -0.272 |               0.211 |           0.467 |               -0.363 |      -0.166 |       0.094 |        0.01  |
| 2026-11-09 | own    | ridge   | {'alpha': 1000.0}         |       190 | 0.03  |                   0.039 |             -0.161 |               0.247 |           0.389 |                0.025 |       0.095 |       0.103 |        0.001 |

