# Forecasts without the filings: backtest (2026-11-10)

Inputs: company statements, leaks, third-party estimates and alternative data, each only from the day it
was public. Never inputs: the S-1 and its amendments (filing = 1) and third-party numbers published after a
filing that may derive from it (filing = 2). Parameters are refit at every vintage. Errors are
log(actual / forecast); negative means the forecast was too high.

## Vintage: publication

| rule        | model                                                    |   n |   mae_log |   bias_log |   mae_log_filing_targets |   mae_log_official_claims |   cov80 |    sd_z |
|:------------|:---------------------------------------------------------|----:|----------:|-----------:|-------------------------:|--------------------------:|--------:|--------:|
| publication | E OpenRouter tokens x price                              |  14 |     0.105 |      0.036 |                    0.117 |                     0.083 |   0.857 |   0.986 |
| publication | E OpenRouter tokens                                      |  14 |     0.114 |      0.017 |                    0.135 |                     0.078 |   0.857 |   1.294 |
| publication | C price-adjusted (github, gamma 1)                       |  14 |     0.155 |     -0.082 |                    0.181 |                     0.107 |   0.643 |   1.58  |
| publication | C price-adjusted (uninformed, gamma 1)                   |  14 |     0.165 |     -0.077 |                    0.174 |                     0.149 |   0.571 |   1.639 |
| publication | Bayes + indicators                                       |  14 |     0.167 |     -0.11  |                    0.205 |                     0.099 |   0.786 |   1.41  |
| publication | C price-adjusted (nov2026_mix, gamma 1)                  |  14 |     0.171 |     -0.083 |                    0.182 |                     0.149 |   0.571 |   1.678 |
| publication | E B + OpenRouter tokens                                  |  14 |     0.173 |     -0.12  |                    0.223 |                     0.084 |   0.714 |   1.755 |
| publication | C price-adjusted (uninformed, gamma 0.5)                 |  14 |     0.184 |     -0.097 |                    0.204 |                     0.149 |   0.571 |   1.832 |
| publication | ML ridge (interp)                                        |  14 |     0.187 |     -0.138 |                    0.189 |                     0.183 | nan     | nan     |
| publication | D + GitHub commits, price-adjusted (uninformed, gamma 1) |  14 |     0.19  |     -0.102 |                    0.209 |                     0.157 |   0.786 |   0.983 |
| publication | B + indicators                                           |  14 |     0.204 |     -0.117 |                    0.235 |                     0.149 |   0.571 |   2.04  |
| publication | A claims only                                            |  14 |     0.207 |     -0.065 |                    0.205 |                     0.211 |   0.643 |   1.499 |
| publication | ML ridge (smoother)                                      |  14 |     0.22  |     -0.178 |                    0.246 |                     0.173 | nan     | nan     |
| publication | D + GitHub commits                                       |  14 |     0.233 |     -0.137 |                    0.276 |                     0.157 |   0.786 |   1.267 |
| publication | E OpenCode downloads                                     |  14 |     0.497 |      0.156 |                    0.536 |                     0.428 |   0.571 |   4.746 |

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
| D + GitHub commits                                       |         48.5 |             95   |            137   |
| D + GitHub commits, price-adjusted (uninformed, gamma 1) |         26.1 |             67.7 |             88.4 |
| E B + OpenRouter tokens                                  |         37.8 |             81.5 |            119.7 |
| E OpenCode downloads                                     |         -0.3 |             58.5 |            100.5 |
| E OpenRouter tokens                                      |        -15.4 |            -12.2 |             58.2 |
| E OpenRouter tokens x price                              |         -5   |            -23   |             33.3 |
| ML ridge (interp)                                        |         40.8 |             63.6 |             81.5 |
| ML ridge (smoother)                                      |         54.4 |             83.8 |            103.2 |

Paired differences in |log error| (negative = second model better; 90% bootstrap interval):

| rule        | comparison                                           |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:------------|:-----------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| publication | B + indicators vs A claims only                      |  15 |      -0.021 |    -0.069 |     0.023 |          6 |       9 |
| publication | C price-adjusted (github, gamma 1) vs B + indicators |  15 |      -0.046 |    -0.075 |    -0.021 |         13 |       1 |
| publication | D + GitHub commits vs B + indicators                 |  15 |       0.079 |     0.001 |     0.18  |          4 |      11 |
| publication | Bayes + indicators vs B + indicators                 |  15 |      -0.03  |    -0.05  |    -0.012 |         12 |       3 |
| publication | E OpenRouter tokens vs A claims only                 |  15 |      -0.081 |    -0.143 |    -0.023 |         10 |       5 |
| publication | E OpenRouter tokens x price vs A claims only         |  15 |      -0.091 |    -0.154 |    -0.029 |         10 |       5 |
| publication | E B + OpenRouter tokens vs B + indicators            |  15 |      -0.029 |    -0.049 |    -0.013 |         14 |       0 |
| publication | E OpenCode downloads vs A claims only                |  15 |       0.558 |     0.168 |     1.073 |          3 |      12 |

## Vintage: period_end

| rule       | model                                                    |   n |   mae_log |   bias_log |   mae_log_filing_targets |   mae_log_official_claims |   cov80 |    sd_z |
|:-----------|:---------------------------------------------------------|----:|----------:|-----------:|-------------------------:|--------------------------:|--------:|--------:|
| period_end | E OpenRouter tokens                                      |  11 |     0.15  |      0.008 |                    0.202 |                     0.087 |   0.636 |   1.695 |
| period_end | E OpenRouter tokens x price                              |  11 |     0.156 |      0.044 |                    0.203 |                     0.099 |   0.636 |   1.175 |
| period_end | C price-adjusted (github, gamma 1)                       |  11 |     0.18  |     -0.143 |                    0.238 |                     0.112 |   0.545 |   1.311 |
| period_end | C price-adjusted (uninformed, gamma 1)                   |  11 |     0.197 |     -0.145 |                    0.236 |                     0.151 |   0.545 |   1.433 |
| period_end | E B + OpenRouter tokens                                  |  11 |     0.208 |     -0.189 |                    0.299 |                     0.099 |   0.636 |   1.519 |
| period_end | A claims only                                            |  11 |     0.224 |     -0.092 |                    0.255 |                     0.187 |   0.727 |   1.366 |
| period_end | C price-adjusted (uninformed, gamma 0.5)                 |  11 |     0.224 |     -0.172 |                    0.285 |                     0.151 |   0.545 |   1.657 |
| period_end | Bayes + indicators                                       |  11 |     0.231 |     -0.22  |                    0.322 |                     0.121 |   0.727 |   1.102 |
| period_end | ML ridge (interp)                                        |  11 |     0.234 |     -0.178 |                    0.273 |                     0.187 | nan     | nan     |
| period_end | B + indicators                                           |  11 |     0.252 |     -0.199 |                    0.335 |                     0.151 |   0.545 |   1.895 |
| period_end | ML ridge (smoother)                                      |  11 |     0.26  |     -0.204 |                    0.323 |                     0.184 | nan     | nan     |
| period_end | D + GitHub commits, price-adjusted (uninformed, gamma 1) |  11 |     0.51  |     -0.494 |                    0.746 |                     0.226 |   0.636 |   0.637 |
| period_end | D + GitHub commits                                       |  11 |     0.568 |     -0.552 |                    0.852 |                     0.226 |   0.545 |   0.875 |
| period_end | E OpenCode downloads                                     |  11 |     0.975 |     -0.901 |                    1.434 |                     0.424 |   0.818 |   1.955 |

H2 2026 targets, % error (+ = forecast too high):

| model                                                    |   rev_2026q3 |   rr_2026_09_s1a |   rr_2026_10_s1a |
|:---------------------------------------------------------|-------------:|-----------------:|-----------------:|
| A claims only                                            |         38.3 |             64   |             71   |
| B + indicators                                           |         39.7 |             79.1 |            115.2 |
| Bayes + indicators                                       |         42.5 |             66.4 |             94.5 |
| C price-adjusted (github, gamma 1)                       |         23   |             43.5 |             74.4 |
| C price-adjusted (uninformed, gamma 0.5)                 |         30.5 |             62.5 |             88.2 |
| C price-adjusted (uninformed, gamma 1)                   |         21.9 |             47.9 |             64.4 |
| D + GitHub commits                                       |         39.1 |             70.4 |            121.5 |
| D + GitHub commits, price-adjusted (uninformed, gamma 1) |         15.6 |             41.1 |             69.9 |
| E B + OpenRouter tokens                                  |         29.9 |             59.8 |            114.8 |
| E OpenCode downloads                                     |         -0.8 |             32.8 |             55.7 |
| E OpenRouter tokens                                      |        -16.4 |            -22.5 |             74.1 |
| E OpenRouter tokens x price                              |        -20.8 |            -31.3 |             49.4 |
| ML ridge (interp)                                        |         40.8 |             63.6 |             81.5 |
| ML ridge (smoother)                                      |         54.4 |             79.2 |             98   |

Paired differences in |log error| (negative = second model better; 90% bootstrap interval):

| rule       | comparison                                           |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:-----------|:-----------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| period_end | B + indicators vs A claims only                      |  13 |      -0.011 |    -0.078 |     0.054 |          5 |       8 |
| period_end | C price-adjusted (github, gamma 1) vs B + indicators |  13 |      -0.06  |    -0.097 |    -0.027 |          9 |       0 |
| period_end | D + GitHub commits vs B + indicators                 |  13 |       0.327 |     0.122 |     0.553 |          6 |       7 |
| period_end | Bayes + indicators vs B + indicators                 |  13 |      -0.01  |    -0.044 |     0.025 |          7 |       6 |
| period_end | E OpenRouter tokens vs A claims only                 |  13 |      -0.052 |    -0.101 |    -0.007 |          7 |       6 |
| period_end | E OpenRouter tokens x price vs A claims only         |  13 |      -0.047 |    -0.089 |    -0.008 |          8 |       5 |
| period_end | E B + OpenRouter tokens vs B + indicators            |  13 |      -0.037 |    -0.066 |    -0.012 |          8 |       3 |
| period_end | E OpenCode downloads vs A claims only                |  13 |       1.366 |     0.502 |     2.316 |          5 |       8 |

