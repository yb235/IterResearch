# Forecasts without the filings: backtest (2026-11-10)

Inputs: company statements, leaks, third-party estimates and alternative data, each only from the day it
was public. Never inputs: the S-1 and its amendments (filing = 1) and third-party numbers published after a
filing that may derive from it (filing = 2). Parameters are refit at every vintage. Errors are
log(actual / forecast); negative means the forecast was too high.

## Vintage: publication

| rule        | model                                                    |   n |   mae_log |   bias_log |   mae_log_filing_targets |   mae_log_official_claims |   cov80 |    sd_z |
|:------------|:---------------------------------------------------------|----:|----------:|-----------:|-------------------------:|--------------------------:|--------:|--------:|
| publication | E OpenRouter tokens x price                              |  15 |     0.132 |     -0.001 |                    0.117 |                     0.155 |   0.8   |   1.049 |
| publication | F core signals x price (selected by backtest)            |  15 |     0.134 |     -0.092 |                    0.167 |                     0.083 |   0.8   |   1.265 |
| publication | E OpenRouter tokens                                      |  15 |     0.141 |     -0.018 |                    0.135 |                     0.151 |   0.8   |   1.316 |
| publication | C price-adjusted (github, gamma 1)                       |  15 |     0.155 |     -0.087 |                    0.181 |                     0.117 |   0.667 |   1.532 |
| publication | C price-adjusted (uninformed, gamma 1)                   |  15 |     0.165 |     -0.083 |                    0.174 |                     0.152 |   0.6   |   1.589 |
| publication | C price-adjusted (nov2026_mix, gamma 1)                  |  15 |     0.17  |     -0.089 |                    0.182 |                     0.152 |   0.6   |   1.626 |
| publication | Bayes + indicators                                       |  15 |     0.171 |     -0.118 |                    0.205 |                     0.12  |   0.8   |   1.365 |
| publication | ML all signals (interp)                                  |  14 |     0.173 |     -0.049 |                    0.143 |                     0.226 | nan     | nan     |
| publication | E B + OpenRouter tokens                                  |  15 |     0.173 |     -0.123 |                    0.223 |                     0.097 |   0.733 |   1.697 |
| publication | C price-adjusted (uninformed, gamma 0.5)                 |  15 |     0.183 |     -0.102 |                    0.204 |                     0.152 |   0.6   |   1.773 |
| publication | ML ridge (interp)                                        |  14 |     0.187 |     -0.138 |                    0.189 |                     0.183 | nan     | nan     |
| publication | ML all signals (smoother)                                |  14 |     0.19  |     -0.129 |                    0.202 |                     0.169 | nan     | nan     |
| publication | Combination: weights from past accuracy (all signals)    |  15 |     0.198 |     -0.132 |                    0.143 |                     0.281 | nan     | nan     |
| publication | B + indicators                                           |  15 |     0.202 |     -0.12  |                    0.235 |                     0.152 |   0.6   |   1.973 |
| publication | Combination: equal weights (all signals)                 |  15 |     0.217 |     -0.11  |                    0.188 |                     0.261 | nan     | nan     |
| publication | ML ridge (smoother)                                      |  14 |     0.22  |     -0.178 |                    0.246 |                     0.173 | nan     | nan     |
| publication | A claims only                                            |  15 |     0.223 |     -0.09  |                    0.205 |                     0.249 |   0.6   |   1.487 |
| publication | Ensemble F + ML (all signals)                            |  14 |     0.235 |     -0.042 |                    0.277 |                     0.161 | nan     | nan     |
| publication | D + GitHub commits, price-adjusted (uninformed, gamma 1) |  15 |     0.241 |     -0.159 |                    0.209 |                     0.289 |   0.733 |   1.309 |
| publication | D + GitHub commits                                       |  15 |     0.281 |     -0.192 |                    0.276 |                     0.289 |   0.733 |   1.502 |
| publication | F all signals x price                                    |  15 |     0.759 |     -0.256 |                    0.55  |                     1.072 |   0.533 |   4.893 |
| publication | E OpenCode downloads                                     |  15 |     0.781 |     -0.171 |                    0.536 |                     1.148 |   0.533 |   4.75  |
| publication | F all signals                                            |  15 |     0.785 |     -0.275 |                    0.597 |                     1.068 |   0.467 |   4.925 |

H2 2026 targets, % error (+ = forecast too high):

| model                                                    |   rev_2026q3 |   rr_2026_09_s1a |   rr_2026_10_s1a |
|:---------------------------------------------------------|-------------:|-----------------:|-----------------:|
| A claims only                                            |         38.3 |             76.5 |             90   |
| B + indicators                                           |         42.3 |             87.5 |            119.8 |
| Bayes + indicators                                       |         35.5 |             73.9 |            100.9 |
| C price-adjusted (github, gamma 1)                       |         29.3 |             59.4 |             78   |
| C price-adjusted (nov2026_mix, gamma 1)                  |         30.5 |             63.6 |             72.3 |
| C price-adjusted (uninformed, gamma 0.5)                 |         33.6 |             72   |             93.6 |
| C price-adjusted (uninformed, gamma 1)                   |         25.3 |             58.2 |             70.8 |
| Combination: equal weights (all signals)                 |         20   |             45.4 |             77.8 |
| Combination: weights from past accuracy (all signals)    |         22.7 |             40.1 |             72.3 |
| D + GitHub commits                                       |         48.5 |             95   |            137   |
| D + GitHub commits, price-adjusted (uninformed, gamma 1) |         26.1 |             67.7 |             88.4 |
| E B + OpenRouter tokens                                  |         37.8 |             81.5 |            119.7 |
| E OpenCode downloads                                     |         -0.3 |             58.5 |            100.5 |
| E OpenRouter tokens                                      |        -15.4 |            -12.2 |             58.2 |
| E OpenRouter tokens x price                              |         -5   |            -23   |             33.3 |
| Ensemble F + ML (all signals)                            |         26.6 |             57.2 |             72.9 |
| F all signals                                            |         33.9 |             65.6 |            157.1 |
| F all signals x price                                    |         19.4 |             47.6 |            116.7 |
| F core signals x price (selected by backtest)            |         24.8 |             52.9 |             77.1 |
| ML all signals (interp)                                  |         26.4 |             63.5 |             33.1 |
| ML all signals (smoother)                                |         34.6 |             61.1 |             79.1 |
| ML ridge (interp)                                        |         40.8 |             63.6 |             81.5 |
| ML ridge (smoother)                                      |         54.4 |             83.8 |            103.2 |

Paired differences in |log error| (negative = second model better; 90% bootstrap interval):

| rule        | comparison                                                             |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:------------|:-----------------------------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| publication | B + indicators vs A claims only                                        |  15 |      -0.021 |    -0.069 |     0.023 |          6 |       9 |
| publication | C price-adjusted (github, gamma 1) vs B + indicators                   |  15 |      -0.046 |    -0.075 |    -0.021 |         13 |       1 |
| publication | D + GitHub commits vs B + indicators                                   |  15 |       0.079 |     0.001 |     0.18  |          4 |      11 |
| publication | Bayes + indicators vs B + indicators                                   |  15 |      -0.03  |    -0.05  |    -0.012 |         12 |       3 |
| publication | E OpenRouter tokens vs A claims only                                   |  15 |      -0.081 |    -0.143 |    -0.023 |         10 |       5 |
| publication | E OpenRouter tokens x price vs A claims only                           |  15 |      -0.091 |    -0.154 |    -0.029 |         10 |       5 |
| publication | E B + OpenRouter tokens vs B + indicators                              |  15 |      -0.029 |    -0.049 |    -0.013 |         14 |       0 |
| publication | E OpenCode downloads vs A claims only                                  |  15 |       0.558 |     0.168 |     1.073 |          3 |      12 |
| publication | F all signals x price vs E OpenRouter tokens x price                   |  15 |       0.627 |     0.258 |     1.096 |          1 |      14 |
| publication | ML all signals (interp) vs ML ridge (interp)                           |  14 |      -0.014 |    -0.092 |     0.061 |          4 |       3 |
| publication | Combination: weights from past accuracy (all signals) vs A claims only |  15 |      -0.024 |    -0.095 |     0.067 |         12 |       3 |

## Vintage: period_end

| rule       | model                                                    |   n |   mae_log |   bias_log |   mae_log_filing_targets |   mae_log_official_claims |   cov80 |    sd_z |
|:-----------|:---------------------------------------------------------|----:|----------:|-----------:|-------------------------:|--------------------------:|--------:|--------:|
| period_end | F core signals x price (selected by backtest)            |  13 |     0.153 |     -0.141 |                    0.198 |                     0.099 |   0.846 |   0.902 |
| period_end | C price-adjusted (github, gamma 1)                       |  13 |     0.176 |     -0.145 |                    0.223 |                     0.121 |   0.615 |   1.211 |
| period_end | C price-adjusted (uninformed, gamma 1)                   |  13 |     0.19  |     -0.146 |                    0.222 |                     0.153 |   0.615 |   1.323 |
| period_end | E OpenRouter tokens                                      |  13 |     0.196 |     -0.062 |                    0.227 |                     0.159 |   0.615 |   1.65  |
| period_end | E B + OpenRouter tokens                                  |  13 |     0.199 |     -0.183 |                    0.276 |                     0.11  |   0.692 |   1.409 |
| period_end | E OpenRouter tokens x price                              |  13 |     0.2   |     -0.031 |                    0.228 |                     0.168 |   0.615 |   1.28  |
| period_end | C price-adjusted (uninformed, gamma 0.5)                 |  13 |     0.213 |     -0.169 |                    0.264 |                     0.153 |   0.615 |   1.53  |
| period_end | Bayes + indicators                                       |  13 |     0.227 |     -0.217 |                    0.302 |                     0.139 |   0.769 |   1.02  |
| period_end | ML ridge (interp)                                        |  11 |     0.234 |     -0.178 |                    0.273 |                     0.187 | nan     | nan     |
| period_end | B + indicators                                           |  13 |     0.236 |     -0.192 |                    0.307 |                     0.153 |   0.615 |   1.752 |
| period_end | ML all signals (smoother)                                |  11 |     0.243 |     -0.159 |                    0.293 |                     0.184 | nan     | nan     |
| period_end | A claims only                                            |  13 |     0.248 |     -0.136 |                    0.264 |                     0.229 |   0.692 |   1.329 |
| period_end | ML all signals (interp)                                  |  11 |     0.25  |     -0.077 |                    0.257 |                     0.243 | nan     | nan     |
| period_end | ML ridge (smoother)                                      |  11 |     0.26  |     -0.204 |                    0.323 |                     0.184 | nan     | nan     |
| period_end | Combination: weights from past accuracy (all signals)    |  13 |     0.305 |     -0.254 |                    0.333 |                     0.272 | nan     | nan     |
| period_end | Combination: equal weights (all signals)                 |  13 |     0.397 |     -0.354 |                    0.491 |                     0.287 | nan     | nan     |
| period_end | D + GitHub commits, price-adjusted (uninformed, gamma 1) |  13 |     0.514 |     -0.501 |                    0.658 |                     0.347 |   0.615 |   1.002 |
| period_end | D + GitHub commits                                       |  13 |     0.564 |     -0.55  |                    0.749 |                     0.347 |   0.538 |   1.111 |
| period_end | Ensemble F + ML (all signals)                            |  11 |     0.57  |     -0.517 |                    0.855 |                     0.229 | nan     | nan     |
| period_end | F all signals x price                                    |  13 |     1.59  |     -1.59  |                    2.008 |                     1.102 |   0.538 |   2.347 |
| period_end | E OpenCode downloads                                     |  13 |     1.614 |     -1.551 |                    2.016 |                     1.145 |   0.769 |   1.902 |
| period_end | F all signals                                            |  13 |     1.624 |     -1.624 |                    2.073 |                     1.1   |   0.385 |   2.315 |

H2 2026 targets, % error (+ = forecast too high):

| model                                                    |   rev_2026q3 |   rr_2026_09_s1a |   rr_2026_10_s1a |
|:---------------------------------------------------------|-------------:|-----------------:|-----------------:|
| A claims only                                            |         38.3 |             64   |             71   |
| B + indicators                                           |         39.7 |             79.1 |            115.2 |
| Bayes + indicators                                       |         42.5 |             66.4 |             94.5 |
| C price-adjusted (github, gamma 1)                       |         23   |             43.5 |             74.4 |
| C price-adjusted (uninformed, gamma 0.5)                 |         30.5 |             62.5 |             88.2 |
| C price-adjusted (uninformed, gamma 1)                   |         21.9 |             47.9 |             64.4 |
| Combination: equal weights (all signals)                 |         15.6 |             33   |             71.9 |
| Combination: weights from past accuracy (all signals)    |         17   |             31.4 |             73.4 |
| D + GitHub commits                                       |         39.1 |             70.4 |            121.5 |
| D + GitHub commits, price-adjusted (uninformed, gamma 1) |         15.6 |             41.1 |             69.9 |
| E B + OpenRouter tokens                                  |         29.9 |             59.8 |            114.8 |
| E OpenCode downloads                                     |         -0.8 |             32.8 |             55.7 |
| E OpenRouter tokens                                      |        -16.4 |            -22.5 |             74.1 |
| E OpenRouter tokens x price                              |        -20.8 |            -31.3 |             49.4 |
| Ensemble F + ML (all signals)                            |         24.2 |             44.8 |             58.7 |
| F all signals                                            |         23.6 |             37.9 |            123.3 |
| F all signals x price                                    |         12.6 |             18.1 |             75.5 |
| F core signals x price (selected by backtest)            |         15.9 |             29.5 |             74.3 |
| ML all signals (interp)                                  |         26.4 |             63.5 |             33.1 |
| ML all signals (smoother)                                |         34.6 |             57.2 |             71.1 |
| ML ridge (interp)                                        |         40.8 |             63.6 |             81.5 |
| ML ridge (smoother)                                      |         54.4 |             79.2 |             98   |

Paired differences in |log error| (negative = second model better; 90% bootstrap interval):

| rule       | comparison                                                             |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:-----------|:-----------------------------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| period_end | B + indicators vs A claims only                                        |  13 |      -0.011 |    -0.078 |     0.054 |          5 |       8 |
| period_end | C price-adjusted (github, gamma 1) vs B + indicators                   |  13 |      -0.06  |    -0.097 |    -0.027 |          9 |       0 |
| period_end | D + GitHub commits vs B + indicators                                   |  13 |       0.327 |     0.122 |     0.553 |          6 |       7 |
| period_end | Bayes + indicators vs B + indicators                                   |  13 |      -0.01  |    -0.044 |     0.025 |          7 |       6 |
| period_end | E OpenRouter tokens vs A claims only                                   |  13 |      -0.052 |    -0.101 |    -0.007 |          7 |       6 |
| period_end | E OpenRouter tokens x price vs A claims only                           |  13 |      -0.047 |    -0.089 |    -0.008 |          8 |       5 |
| period_end | E B + OpenRouter tokens vs B + indicators                              |  13 |      -0.037 |    -0.066 |    -0.012 |          8 |       3 |
| period_end | E OpenCode downloads vs A claims only                                  |  13 |       1.366 |     0.502 |     2.316 |          5 |       8 |
| period_end | F all signals x price vs E OpenRouter tokens x price                   |  13 |       1.389 |     0.643 |     2.184 |          3 |      10 |
| period_end | ML all signals (interp) vs ML ridge (interp)                           |  11 |       0.016 |    -0.092 |     0.116 |          4 |       7 |
| period_end | Combination: weights from past accuracy (all signals) vs A claims only |  13 |       0.057 |    -0.061 |     0.197 |          5 |       8 |

