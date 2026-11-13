# 'Over $X' claims as an interval (sub-task N): backtest (2026-11-13)

Pre-registered in DATA_PLAN.md (子任务 N, commit abce18a) before any full rerun. Primary rule **iv115**: a
`kind = lower` claim X means X <= true value <= 1.15 X (two-sided truncated-normal update in model.py,
`config.LOWER_UPPER_RATIO`). Variants reported regardless of outcome: baseline (one-sided, the original rule),
point (a point claim at X), iv110, iv125. Inputs never include the filings; parameters refit at every vintage;
crossing position profiled. Errors are log(actual / forecast); + = forecast too low.

Disclosure (from DATA_PLAN): before registering the rule, the orchestrator saw baseline / point / point-at-1.05X
for model A on rr_2026_05, rev_2026q3, rr_2026_09_s1a and rr_2026_10_s1a (publication rule).

- publication: 14 common targets (rev_2025fy, rev_2025q1, rev_2025q2, rev_2025q3, rev_2026q1, rev_2026q2, rev_2026q3, rr_2025_12, rr_2026_02, rr_2026_03, rr_2026_04, rr_2026_05, rr_2026_09_s1a, rr_2026_10_s1a); post-t0: rev_2026q3, rr_2026_02, rr_2026_03, rr_2026_04, rr_2026_05, rr_2026_09_s1a, rr_2026_10_s1a.
- period_end: 11 common targets (rev_2025fy, rev_2026q1, rev_2026q2, rev_2026q3, rr_2025_12, rr_2026_02, rr_2026_03, rr_2026_04, rr_2026_05, rr_2026_09_s1a, rr_2026_10_s1a); post-t0: rev_2025fy, rev_2026q1, rev_2026q2, rev_2026q3, rr_2025_12, rr_2026_02, rr_2026_03, rr_2026_04, rr_2026_05, rr_2026_09_s1a, rr_2026_10_s1a.
- Variants run: A claims only: baseline, point, iv110, iv115, iv125; B + indicators: baseline, point, iv110, iv115, iv125; E OpenRouter tokens x price: baseline, point, iv110, iv115, iv125; G GitHub new-repo share: baseline, point, iv110, iv115, iv125; R Ramp adoption: baseline, point, iv110, iv115, iv125; Bayes + indicators: baseline, iv115; M1 (pooled): baseline, point, iv110, iv115, iv125; M2 (pooled): baseline, point, iv110, iv115, iv125; M1 (own): baseline, point, iv110, iv115, iv125; M2 (own): baseline, point, iv110, iv115, iv125; SS claims-only path (ml_fixed): baseline, point, iv110, iv115, iv125.

## Mean |log error| by model x variant x rule (common targets of every model and variant)

| rule        | model                          | variant   |   n_all |   mae_all |   bias_all |   n_post |   mae_post |   bias_post |   n_h2 |   mae_h2 |   cov80 |   cov80_post |
|:------------|:-------------------------------|:----------|--------:|----------:|-----------:|---------:|-----------:|------------:|-------:|---------:|--------:|-------------:|
| period_end  | A claims only                  | baseline  |      11 |     0.224 |     -0.092 |       11 |      0.224 |      -0.092 |      3 |    0.452 |   0.727 |        0.727 |
| period_end  | A claims only                  | point     |      11 |     0.144 |     -0.02  |       11 |      0.144 |      -0.02  |      3 |    0.182 |   0.818 |        0.818 |
| period_end  | A claims only                  | iv110     |      11 |     0.165 |     -0.037 |       11 |      0.165 |      -0.037 |      3 |    0.249 |   0.818 |        0.818 |
| period_end  | A claims only                  | iv115     |      11 |     0.175 |     -0.045 |       11 |      0.175 |      -0.045 |      3 |    0.28  |   0.818 |        0.818 |
| period_end  | A claims only                  | iv125     |      11 |     0.192 |     -0.059 |       11 |      0.192 |      -0.059 |      3 |    0.334 |   0.727 |        0.727 |
| period_end  | B + indicators                 | baseline  |      11 |     0.252 |     -0.199 |       11 |      0.252 |      -0.199 |      3 |    0.561 |   0.545 |        0.545 |
| period_end  | B + indicators                 | point     |      11 |     0.207 |     -0.161 |       11 |      0.207 |      -0.161 |      3 |    0.431 |   0.545 |        0.545 |
| period_end  | B + indicators                 | iv110     |      11 |     0.221 |     -0.172 |       11 |      0.221 |      -0.172 |      3 |    0.467 |   0.545 |        0.545 |
| period_end  | B + indicators                 | iv115     |      11 |     0.227 |     -0.178 |       11 |      0.227 |      -0.178 |      3 |    0.484 |   0.545 |        0.545 |
| period_end  | B + indicators                 | iv125     |      11 |     0.237 |     -0.189 |       11 |      0.237 |      -0.189 |      3 |    0.514 |   0.545 |        0.545 |
| period_end  | E OpenRouter tokens x price    | baseline  |      11 |     0.139 |      0.026 |       11 |      0.139 |       0.026 |      3 |    0.262 |   0.909 |        0.909 |
| period_end  | E OpenRouter tokens x price    | point     |      11 |     0.124 |      0.037 |       11 |      0.124 |       0.037 |      3 |    0.241 |   0.727 |        0.727 |
| period_end  | E OpenRouter tokens x price    | iv110     |      11 |     0.123 |      0.024 |       11 |      0.123 |       0.024 |      3 |    0.234 |   0.909 |        0.909 |
| period_end  | E OpenRouter tokens x price    | iv115     |      11 |     0.123 |      0.02  |       11 |      0.123 |       0.02  |      3 |    0.233 |   0.909 |        0.909 |
| period_end  | E OpenRouter tokens x price    | iv125     |      11 |     0.123 |      0.015 |       11 |      0.123 |       0.015 |      3 |    0.23  |   0.909 |        0.909 |
| period_end  | G GitHub new-repo share        | baseline  |      11 |     0.151 |      0.125 |       11 |      0.151 |       0.125 |      3 |    0.047 |   0.818 |        0.818 |
| period_end  | G GitHub new-repo share        | point     |      11 |     0.113 |      0.112 |       11 |      0.113 |       0.112 |      3 |    0.069 |   0.909 |        0.909 |
| period_end  | G GitHub new-repo share        | iv110     |      11 |     0.108 |      0.1   |       11 |      0.108 |       0.1   |      3 |    0.052 |   0.909 |        0.909 |
| period_end  | G GitHub new-repo share        | iv115     |      11 |     0.107 |      0.095 |       11 |      0.107 |       0.095 |      3 |    0.047 |   0.909 |        0.909 |
| period_end  | G GitHub new-repo share        | iv125     |      11 |     0.108 |      0.091 |       11 |      0.108 |       0.091 |      3 |    0.041 |   0.818 |        0.818 |
| period_end  | R Ramp adoption                | baseline  |      11 |     0.506 |      0.482 |       11 |      0.506 |       0.482 |      3 |    0.046 |   0.636 |        0.636 |
| period_end  | R Ramp adoption                | point     |      11 |     0.387 |      0.374 |       11 |      0.387 |       0.374 |      3 |    0.054 |   0.636 |        0.636 |
| period_end  | R Ramp adoption                | iv110     |      11 |     0.39  |      0.371 |       11 |      0.39  |       0.371 |      3 |    0.047 |   0.636 |        0.636 |
| period_end  | R Ramp adoption                | iv115     |      11 |     0.393 |      0.372 |       11 |      0.393 |       0.372 |      3 |    0.044 |   0.636 |        0.636 |
| period_end  | R Ramp adoption                | iv125     |      11 |     0.398 |      0.375 |       11 |      0.398 |       0.375 |      3 |    0.042 |   0.636 |        0.636 |
| period_end  | Bayes + indicators             | baseline  |      11 |     0.231 |     -0.22  |       11 |      0.231 |      -0.22  |      3 |    0.509 |   0.727 |        0.727 |
| period_end  | Bayes + indicators             | iv115     |      11 |     0.207 |     -0.187 |       11 |      0.207 |      -0.187 |      3 |    0.439 |   0.727 |        0.727 |
| period_end  | M1 (pooled)                    | baseline  |      11 |     0.286 |      0.271 |       11 |      0.286 |       0.271 |      3 |    0.726 | nan     |      nan     |
| period_end  | M1 (pooled)                    | point     |      11 |     0.32  |      0.305 |       11 |      0.32  |       0.305 |      3 |    0.85  | nan     |      nan     |
| period_end  | M1 (pooled)                    | iv110     |      11 |     0.312 |      0.296 |       11 |      0.312 |       0.296 |      3 |    0.819 | nan     |      nan     |
| period_end  | M1 (pooled)                    | iv115     |      11 |     0.308 |      0.292 |       11 |      0.308 |       0.292 |      3 |    0.804 | nan     |      nan     |
| period_end  | M1 (pooled)                    | iv125     |      11 |     0.301 |      0.285 |       11 |      0.301 |       0.285 |      3 |    0.779 | nan     |      nan     |
| period_end  | M2 (pooled)                    | baseline  |      11 |     0.122 |      0.09  |       11 |      0.122 |       0.09  |      3 |    0.137 | nan     |      nan     |
| period_end  | M2 (pooled)                    | point     |      11 |     0.173 |      0.142 |       11 |      0.173 |       0.142 |      3 |    0.334 | nan     |      nan     |
| period_end  | M2 (pooled)                    | iv110     |      11 |     0.16  |      0.13  |       11 |      0.16  |       0.13  |      3 |    0.285 | nan     |      nan     |
| period_end  | M2 (pooled)                    | iv115     |      11 |     0.154 |      0.123 |       11 |      0.154 |       0.123 |      3 |    0.262 | nan     |      nan     |
| period_end  | M2 (pooled)                    | iv125     |      11 |     0.144 |      0.113 |       11 |      0.144 |       0.113 |      3 |    0.222 | nan     |      nan     |
| period_end  | M1 (own)                       | baseline  |      11 |     0.232 |      0.204 |       11 |      0.232 |       0.204 |      3 |    0.573 | nan     |      nan     |
| period_end  | M1 (own)                       | point     |      11 |     0.266 |      0.241 |       11 |      0.266 |       0.241 |      3 |    0.698 | nan     |      nan     |
| period_end  | M1 (own)                       | iv110     |      11 |     0.258 |      0.232 |       11 |      0.258 |       0.232 |      3 |    0.666 | nan     |      nan     |
| period_end  | M1 (own)                       | iv115     |      11 |     0.254 |      0.227 |       11 |      0.254 |       0.227 |      3 |    0.652 | nan     |      nan     |
| period_end  | M1 (own)                       | iv125     |      11 |     0.248 |      0.22  |       11 |      0.248 |       0.22  |      3 |    0.626 | nan     |      nan     |
| period_end  | M2 (own)                       | baseline  |      11 |     0.094 |      0.056 |       11 |      0.094 |       0.056 |      3 |    0.061 | nan     |      nan     |
| period_end  | M2 (own)                       | point     |      11 |     0.146 |      0.11  |       11 |      0.146 |       0.11  |      3 |    0.258 | nan     |      nan     |
| period_end  | M2 (own)                       | iv110     |      11 |     0.134 |      0.097 |       11 |      0.134 |       0.097 |      3 |    0.209 | nan     |      nan     |
| period_end  | M2 (own)                       | iv115     |      11 |     0.128 |      0.091 |       11 |      0.128 |       0.091 |      3 |    0.186 | nan     |      nan     |
| period_end  | M2 (own)                       | iv125     |      11 |     0.117 |      0.08  |       11 |      0.117 |       0.08  |      3 |    0.146 | nan     |      nan     |
| period_end  | SS claims-only path (ml_fixed) | baseline  |      11 |     0.224 |     -0.092 |       11 |      0.224 |      -0.092 |      3 |    0.452 | nan     |      nan     |
| period_end  | SS claims-only path (ml_fixed) | point     |      11 |     0.144 |     -0.02  |       11 |      0.144 |      -0.02  |      3 |    0.182 | nan     |      nan     |
| period_end  | SS claims-only path (ml_fixed) | iv110     |      11 |     0.165 |     -0.037 |       11 |      0.165 |      -0.037 |      3 |    0.249 | nan     |      nan     |
| period_end  | SS claims-only path (ml_fixed) | iv115     |      11 |     0.175 |     -0.045 |       11 |      0.175 |      -0.045 |      3 |    0.28  | nan     |      nan     |
| period_end  | SS claims-only path (ml_fixed) | iv125     |      11 |     0.192 |     -0.059 |       11 |      0.192 |      -0.059 |      3 |    0.335 | nan     |      nan     |
| period_end  | N naive last claim             | -         |      11 |     0.257 |      0.25  |       11 |      0.257 |       0.25  |      3 |    0.056 | nan     |      nan     |
| publication | A claims only                  | baseline  |      14 |     0.207 |     -0.065 |        7 |      0.347 |      -0.151 |      3 |    0.512 |   0.643 |        0.429 |
| publication | A claims only                  | point     |      14 |     0.107 |      0.019 |        7 |      0.159 |       0.029 |      3 |    0.085 |   0.786 |        0.714 |
| publication | A claims only                  | iv110     |      14 |     0.118 |      0.009 |        7 |      0.183 |       0.011 |      3 |    0.134 |   0.786 |        0.714 |
| publication | A claims only                  | iv115     |      14 |     0.124 |      0.004 |        7 |      0.193 |       0.001 |      3 |    0.157 |   0.714 |        0.571 |
| publication | A claims only                  | iv125     |      14 |     0.134 |     -0.004 |        7 |      0.213 |      -0.016 |      3 |    0.199 |   0.714 |        0.571 |
| publication | B + indicators                 | baseline  |      14 |     0.204 |     -0.117 |        7 |      0.345 |      -0.266 |      3 |    0.59  |   0.571 |        0.286 |
| publication | B + indicators                 | point     |      14 |     0.13  |     -0.057 |        7 |      0.203 |      -0.134 |      3 |    0.282 |   0.571 |        0.286 |
| publication | B + indicators                 | iv110     |      14 |     0.14  |     -0.064 |        7 |      0.222 |      -0.149 |      3 |    0.318 |   0.571 |        0.286 |
| publication | B + indicators                 | iv115     |      14 |     0.144 |     -0.067 |        7 |      0.231 |      -0.157 |      3 |    0.336 |   0.571 |        0.286 |
| publication | B + indicators                 | iv125     |      14 |     0.153 |     -0.074 |        7 |      0.246 |      -0.175 |      3 |    0.369 |   0.571 |        0.286 |
| publication | E OpenRouter tokens x price    | baseline  |      14 |     0.104 |      0.037 |        7 |      0.142 |       0.059 |      3 |    0.195 |   0.857 |        0.714 |
| publication | E OpenRouter tokens x price    | point     |      14 |     0.096 |      0.066 |        7 |      0.132 |       0.096 |      3 |    0.185 |   0.929 |        0.857 |
| publication | E OpenRouter tokens x price    | iv110     |      14 |     0.093 |      0.059 |        7 |      0.13  |       0.083 |      3 |    0.181 |   0.857 |        0.714 |
| publication | E OpenRouter tokens x price    | iv115     |      14 |     0.094 |      0.059 |        7 |      0.13  |       0.078 |      3 |    0.179 |   0.857 |        0.714 |
| publication | E OpenRouter tokens x price    | iv125     |      14 |     0.074 |      0.042 |        7 |      0.09  |       0.045 |      3 |    0.084 |   1     |        1     |
| publication | G GitHub new-repo share        | baseline  |      14 |     0.099 |     -0.012 |        7 |      0.171 |       0.001 |      3 |    0.203 |   0.857 |        0.714 |
| publication | G GitHub new-repo share        | point     |      14 |     0.064 |      0.023 |        7 |      0.086 |       0.073 |      3 |    0.043 |   0.929 |        0.857 |
| publication | G GitHub new-repo share        | iv110     |      14 |     0.063 |      0.015 |        7 |      0.091 |       0.057 |      3 |    0.051 |   0.929 |        0.857 |
| publication | G GitHub new-repo share        | iv115     |      14 |     0.064 |      0.011 |        7 |      0.095 |       0.05  |      3 |    0.061 |   0.929 |        0.857 |
| publication | G GitHub new-repo share        | iv125     |      14 |     0.068 |      0.006 |        7 |      0.106 |       0.039 |      3 |    0.08  |   0.929 |        0.857 |
| publication | R Ramp adoption                | baseline  |      14 |     0.137 |      0.077 |        7 |      0.224 |       0.123 |      3 |    0.119 |   0.857 |        0.714 |
| publication | R Ramp adoption                | point     |      14 |     0.111 |      0.078 |        7 |      0.153 |       0.14  |      3 |    0.045 |   0.929 |        0.857 |
| publication | R Ramp adoption                | iv110     |      14 |     0.108 |      0.071 |        7 |      0.151 |       0.125 |      3 |    0.039 |   0.929 |        0.857 |
| publication | R Ramp adoption                | iv115     |      14 |     0.11  |      0.068 |        7 |      0.155 |       0.119 |      3 |    0.049 |   0.929 |        0.857 |
| publication | R Ramp adoption                | iv125     |      14 |     0.114 |      0.065 |        7 |      0.166 |       0.112 |      3 |    0.066 |   0.929 |        0.857 |
| publication | Bayes + indicators             | baseline  |      14 |     0.167 |     -0.11  |        7 |      0.283 |      -0.251 |      3 |    0.518 |   0.786 |        0.571 |
| publication | Bayes + indicators             | iv115     |      14 |     0.121 |     -0.065 |        7 |      0.187 |      -0.155 |      3 |    0.276 |   0.786 |        0.571 |
| publication | M1 (pooled)                    | baseline  |      14 |     0.245 |      0.221 |        7 |      0.42  |       0.42  |      3 |    0.704 | nan     |      nan     |
| publication | M1 (pooled)                    | point     |      14 |     0.269 |      0.246 |        7 |      0.481 |       0.481 |      3 |    0.85  | nan     |      nan     |
| publication | M1 (pooled)                    | iv110     |      14 |     0.263 |      0.239 |        7 |      0.47  |       0.47  |      3 |    0.821 | nan     |      nan     |
| publication | M1 (pooled)                    | iv115     |      14 |     0.26  |      0.237 |        7 |      0.464 |       0.464 |      3 |    0.808 | nan     |      nan     |
| publication | M1 (pooled)                    | iv125     |      14 |     0.256 |      0.233 |        7 |      0.455 |       0.455 |      3 |    0.786 | nan     |      nan     |
| publication | M2 (pooled)                    | baseline  |      14 |     0.118 |      0.078 |        7 |      0.168 |       0.134 |      3 |    0.126 | nan     |      nan     |
| publication | M2 (pooled)                    | point     |      14 |     0.165 |      0.132 |        7 |      0.274 |       0.255 |      3 |    0.383 | nan     |      nan     |
| publication | M2 (pooled)                    | iv110     |      14 |     0.157 |      0.124 |        7 |      0.259 |       0.24  |      3 |    0.344 | nan     |      nan     |
| publication | M2 (pooled)                    | iv115     |      14 |     0.153 |      0.121 |        7 |      0.251 |       0.233 |      3 |    0.325 | nan     |      nan     |
| publication | M2 (pooled)                    | iv125     |      14 |     0.147 |      0.115 |        7 |      0.238 |       0.219 |      3 |    0.293 | nan     |      nan     |
| publication | M1 (own)                       | baseline  |      14 |     0.199 |      0.175 |        7 |      0.328 |       0.328 |      3 |    0.56  | nan     |      nan     |
| publication | M1 (own)                       | point     |      14 |     0.225 |      0.201 |        7 |      0.393 |       0.393 |      3 |    0.706 | nan     |      nan     |
| publication | M1 (own)                       | iv110     |      14 |     0.218 |      0.195 |        7 |      0.38  |       0.38  |      3 |    0.677 | nan     |      nan     |
| publication | M1 (own)                       | iv115     |      14 |     0.215 |      0.192 |        7 |      0.374 |       0.374 |      3 |    0.664 | nan     |      nan     |
| publication | M1 (own)                       | iv125     |      14 |     0.211 |      0.188 |        7 |      0.365 |       0.365 |      3 |    0.642 | nan     |      nan     |
| publication | M2 (own)                       | baseline  |      14 |     0.096 |      0.055 |        7 |      0.123 |       0.089 |      3 |    0.054 | nan     |      nan     |
| publication | M2 (own)                       | point     |      14 |     0.143 |      0.11  |        7 |      0.231 |       0.211 |      3 |    0.311 | nan     |      nan     |
| publication | M2 (own)                       | iv110     |      14 |     0.135 |      0.102 |        7 |      0.215 |       0.196 |      3 |    0.272 | nan     |      nan     |
| publication | M2 (own)                       | iv115     |      14 |     0.131 |      0.098 |        7 |      0.208 |       0.188 |      3 |    0.253 | nan     |      nan     |
| publication | M2 (own)                       | iv125     |      14 |     0.126 |      0.092 |        7 |      0.195 |       0.175 |      3 |    0.221 | nan     |      nan     |
| publication | SS claims-only path (ml_fixed) | baseline  |      14 |     0.207 |     -0.065 |        7 |      0.347 |      -0.151 |      3 |    0.512 | nan     |      nan     |
| publication | SS claims-only path (ml_fixed) | point     |      14 |     0.107 |      0.019 |        7 |      0.159 |       0.029 |      3 |    0.085 | nan     |      nan     |
| publication | SS claims-only path (ml_fixed) | iv110     |      14 |     0.118 |      0.009 |        7 |      0.183 |       0.011 |      3 |    0.134 | nan     |      nan     |
| publication | SS claims-only path (ml_fixed) | iv115     |      14 |     0.124 |      0.004 |        7 |      0.193 |       0.001 |      3 |    0.157 | nan     |      nan     |
| publication | SS claims-only path (ml_fixed) | iv125     |      14 |     0.134 |     -0.004 |        7 |      0.213 |      -0.016 |      3 |    0.199 | nan     |      nan     |
| publication | N naive last claim             | -         |      14 |     0.221 |      0.215 |        7 |      0.296 |       0.285 |      3 |    0.056 | nan     |      nan     |

## A, E, G, R and naive only (their common targets: all state-space targets)

| rule        | model                       | variant   |   n_all |   mae_all |   bias_all |   n_post |   mae_post |   n_h2 |   mae_h2 |   cov80 |   cov50 |
|:------------|:----------------------------|:----------|--------:|----------:|-----------:|---------:|-----------:|-------:|---------:|--------:|--------:|
| period_end  | A claims only               | baseline  |      13 |     0.248 |     -0.136 |       13 |      0.248 |      3 |    0.452 |   0.692 |   0.308 |
| period_end  | A claims only               | point     |      13 |     0.123 |     -0.019 |       13 |      0.123 |      3 |    0.182 |   0.846 |   0.538 |
| period_end  | A claims only               | iv110     |      13 |     0.153 |     -0.044 |       13 |      0.153 |      3 |    0.249 |   0.846 |   0.462 |
| period_end  | A claims only               | iv115     |      13 |     0.165 |     -0.055 |       13 |      0.165 |      3 |    0.28  |   0.846 |   0.385 |
| period_end  | A claims only               | iv125     |      13 |     0.188 |     -0.076 |       13 |      0.188 |      3 |    0.334 |   0.769 |   0.385 |
| period_end  | E OpenRouter tokens x price | baseline  |      13 |     0.186 |     -0.047 |       13 |      0.186 |      3 |    0.262 |   0.846 |   0.385 |
| period_end  | E OpenRouter tokens x price | point     |      13 |     0.107 |      0.029 |       13 |      0.107 |      3 |    0.241 |   0.769 |   0.615 |
| period_end  | E OpenRouter tokens x price | iv110     |      13 |     0.117 |      0.008 |       13 |      0.117 |      3 |    0.234 |   0.923 |   0.615 |
| period_end  | E OpenRouter tokens x price | iv115     |      13 |     0.121 |     -0     |       13 |      0.121 |      3 |    0.233 |   0.923 |   0.538 |
| period_end  | E OpenRouter tokens x price | iv125     |      13 |     0.13  |     -0.013 |       13 |      0.13  |      3 |    0.23  |   0.923 |   0.538 |
| period_end  | G GitHub new-repo share     | baseline  |      13 |     0.156 |      0.078 |       13 |      0.156 |      3 |    0.047 |   0.846 |   0.692 |
| period_end  | G GitHub new-repo share     | point     |      13 |     0.123 |      0.122 |       13 |      0.123 |      3 |    0.069 |   0.923 |   0.692 |
| period_end  | G GitHub new-repo share     | iv110     |      13 |     0.106 |      0.094 |       13 |      0.106 |      3 |    0.052 |   0.923 |   0.769 |
| period_end  | G GitHub new-repo share     | iv115     |      13 |     0.104 |      0.086 |       13 |      0.104 |      3 |    0.047 |   0.923 |   0.769 |
| period_end  | G GitHub new-repo share     | iv125     |      13 |     0.103 |      0.074 |       13 |      0.103 |      3 |    0.041 |   0.846 |   0.769 |
| period_end  | R Ramp adoption             | baseline  |      13 |     0.524 |      0.475 |       13 |      0.524 |      3 |    0.046 |   0.615 |   0.462 |
| period_end  | R Ramp adoption             | point     |      13 |     0.42  |      0.409 |       13 |      0.42  |      3 |    0.054 |   0.615 |   0.462 |
| period_end  | R Ramp adoption             | iv110     |      13 |     0.42  |      0.404 |       13 |      0.42  |      3 |    0.047 |   0.615 |   0.462 |
| period_end  | R Ramp adoption             | iv115     |      13 |     0.421 |      0.403 |       13 |      0.421 |      3 |    0.044 |   0.615 |   0.462 |
| period_end  | R Ramp adoption             | iv125     |      13 |     0.423 |      0.403 |       13 |      0.423 |      3 |    0.042 |   0.615 |   0.462 |
| period_end  | N naive last claim          | -         |      13 |     0.329 |      0.323 |       13 |      0.329 |      3 |    0.056 | nan     | nan     |
| publication | A claims only               | baseline  |      15 |     0.223 |     -0.09  |        8 |      0.358 |      3 |    0.512 |   0.6   |   0.4   |
| publication | A claims only               | point     |      15 |     0.101 |      0.017 |        8 |      0.141 |      3 |    0.085 |   0.8   |   0.533 |
| publication | A claims only               | iv110     |      15 |     0.117 |      0.002 |        8 |      0.172 |      3 |    0.134 |   0.8   |   0.533 |
| publication | A claims only               | iv115     |      15 |     0.124 |     -0.005 |        8 |      0.186 |      3 |    0.157 |   0.733 |   0.467 |
| publication | A claims only               | iv125     |      15 |     0.138 |     -0.017 |        8 |      0.211 |      3 |    0.199 |   0.733 |   0.4   |
| publication | E OpenRouter tokens x price | baseline  |      15 |     0.131 |     -0     |        8 |      0.189 |      3 |    0.195 |   0.8   |   0.4   |
| publication | E OpenRouter tokens x price | point     |      15 |     0.091 |      0.061 |        8 |      0.117 |      3 |    0.185 |   0.933 |   0.4   |
| publication | E OpenRouter tokens x price | iv110     |      15 |     0.094 |      0.049 |        8 |      0.126 |      3 |    0.181 |   0.867 |   0.533 |
| publication | E OpenRouter tokens x price | iv115     |      15 |     0.097 |      0.046 |        8 |      0.13  |      3 |    0.179 |   0.867 |   0.467 |
| publication | E OpenRouter tokens x price | iv125     |      15 |     0.082 |      0.026 |        8 |      0.104 |      3 |    0.084 |   1     |   0.533 |
| publication | G GitHub new-repo share     | baseline  |      15 |     0.1   |     -0.019 |        8 |      0.164 |      3 |    0.203 |   0.867 |   0.733 |
| publication | G GitHub new-repo share     | point     |      15 |     0.072 |      0.034 |        8 |      0.1   |      3 |    0.043 |   0.933 |   0.733 |
| publication | G GitHub new-repo share     | iv110     |      15 |     0.069 |      0.024 |        8 |      0.098 |      3 |    0.051 |   0.933 |   0.8   |
| publication | G GitHub new-repo share     | iv115     |      15 |     0.069 |      0.019 |        8 |      0.1   |      3 |    0.061 |   0.933 |   0.733 |
| publication | G GitHub new-repo share     | iv125     |      15 |     0.067 |      0.009 |        8 |      0.1   |      3 |    0.08  |   0.933 |   0.733 |
| publication | R Ramp adoption             | baseline  |      15 |     0.14  |      0.059 |        8 |      0.22  |      3 |    0.119 |   0.867 |   0.667 |
| publication | R Ramp adoption             | point     |      15 |     0.118 |      0.088 |        8 |      0.162 |      3 |    0.045 |   0.933 |   0.733 |
| publication | R Ramp adoption             | iv110     |      15 |     0.113 |      0.078 |        8 |      0.154 |      3 |    0.039 |   0.933 |   0.733 |
| publication | R Ramp adoption             | iv115     |      15 |     0.113 |      0.073 |        8 |      0.155 |      3 |    0.049 |   0.933 |   0.733 |
| publication | R Ramp adoption             | iv125     |      15 |     0.115 |      0.068 |        8 |      0.16  |      3 |    0.066 |   0.933 |   0.733 |
| publication | N naive last claim          | -         |      15 |     0.262 |      0.257 |        8 |      0.365 |      3 |    0.056 | nan     | nan     |

## Each model on its own targets

| rule        | model                          | variant   |   n_all |   mae_all |   n_post |   mae_post |   cov80 |
|:------------|:-------------------------------|:----------|--------:|----------:|---------:|-----------:|--------:|
| period_end  | A claims only                  | baseline  |      13 |     0.248 |       13 |      0.248 |   0.692 |
| period_end  | A claims only                  | point     |      13 |     0.123 |       13 |      0.123 |   0.846 |
| period_end  | A claims only                  | iv110     |      13 |     0.153 |       13 |      0.153 |   0.846 |
| period_end  | A claims only                  | iv115     |      13 |     0.165 |       13 |      0.165 |   0.846 |
| period_end  | A claims only                  | iv125     |      13 |     0.188 |       13 |      0.188 |   0.769 |
| period_end  | B + indicators                 | baseline  |      13 |     0.236 |       13 |      0.236 |   0.615 |
| period_end  | B + indicators                 | point     |      13 |     0.178 |       13 |      0.178 |   0.615 |
| period_end  | B + indicators                 | iv110     |      13 |     0.193 |       13 |      0.193 |   0.615 |
| period_end  | B + indicators                 | iv115     |      13 |     0.202 |       13 |      0.202 |   0.615 |
| period_end  | B + indicators                 | iv125     |      13 |     0.215 |       13 |      0.215 |   0.615 |
| period_end  | E OpenRouter tokens x price    | baseline  |      13 |     0.186 |       13 |      0.186 |   0.846 |
| period_end  | E OpenRouter tokens x price    | point     |      13 |     0.107 |       13 |      0.107 |   0.769 |
| period_end  | E OpenRouter tokens x price    | iv110     |      13 |     0.117 |       13 |      0.117 |   0.923 |
| period_end  | E OpenRouter tokens x price    | iv115     |      13 |     0.121 |       13 |      0.121 |   0.923 |
| period_end  | E OpenRouter tokens x price    | iv125     |      13 |     0.13  |       13 |      0.13  |   0.923 |
| period_end  | G GitHub new-repo share        | baseline  |      13 |     0.156 |       13 |      0.156 |   0.846 |
| period_end  | G GitHub new-repo share        | point     |      13 |     0.123 |       13 |      0.123 |   0.923 |
| period_end  | G GitHub new-repo share        | iv110     |      13 |     0.106 |       13 |      0.106 |   0.923 |
| period_end  | G GitHub new-repo share        | iv115     |      13 |     0.104 |       13 |      0.104 |   0.923 |
| period_end  | G GitHub new-repo share        | iv125     |      13 |     0.103 |       13 |      0.103 |   0.846 |
| period_end  | R Ramp adoption                | baseline  |      13 |     0.524 |       13 |      0.524 |   0.615 |
| period_end  | R Ramp adoption                | point     |      13 |     0.42  |       13 |      0.42  |   0.615 |
| period_end  | R Ramp adoption                | iv110     |      13 |     0.42  |       13 |      0.42  |   0.615 |
| period_end  | R Ramp adoption                | iv115     |      13 |     0.421 |       13 |      0.421 |   0.615 |
| period_end  | R Ramp adoption                | iv125     |      13 |     0.423 |       13 |      0.423 |   0.615 |
| period_end  | Bayes + indicators             | baseline  |      13 |     0.227 |       13 |      0.227 |   0.769 |
| period_end  | Bayes + indicators             | iv115     |      13 |     0.184 |       13 |      0.184 |   0.769 |
| period_end  | M1 (pooled)                    | baseline  |      11 |     0.286 |       11 |      0.286 | nan     |
| period_end  | M1 (pooled)                    | point     |      11 |     0.32  |       11 |      0.32  | nan     |
| period_end  | M1 (pooled)                    | iv110     |      11 |     0.312 |       11 |      0.312 | nan     |
| period_end  | M1 (pooled)                    | iv115     |      11 |     0.308 |       11 |      0.308 | nan     |
| period_end  | M1 (pooled)                    | iv125     |      11 |     0.301 |       11 |      0.301 | nan     |
| period_end  | M2 (pooled)                    | baseline  |      11 |     0.122 |       11 |      0.122 | nan     |
| period_end  | M2 (pooled)                    | point     |      11 |     0.173 |       11 |      0.173 | nan     |
| period_end  | M2 (pooled)                    | iv110     |      11 |     0.16  |       11 |      0.16  | nan     |
| period_end  | M2 (pooled)                    | iv115     |      11 |     0.154 |       11 |      0.154 | nan     |
| period_end  | M2 (pooled)                    | iv125     |      11 |     0.144 |       11 |      0.144 | nan     |
| period_end  | M1 (own)                       | baseline  |      11 |     0.232 |       11 |      0.232 | nan     |
| period_end  | M1 (own)                       | point     |      11 |     0.266 |       11 |      0.266 | nan     |
| period_end  | M1 (own)                       | iv110     |      11 |     0.258 |       11 |      0.258 | nan     |
| period_end  | M1 (own)                       | iv115     |      11 |     0.254 |       11 |      0.254 | nan     |
| period_end  | M1 (own)                       | iv125     |      11 |     0.248 |       11 |      0.248 | nan     |
| period_end  | M2 (own)                       | baseline  |      11 |     0.094 |       11 |      0.094 | nan     |
| period_end  | M2 (own)                       | point     |      11 |     0.146 |       11 |      0.146 | nan     |
| period_end  | M2 (own)                       | iv110     |      11 |     0.134 |       11 |      0.134 | nan     |
| period_end  | M2 (own)                       | iv115     |      11 |     0.128 |       11 |      0.128 | nan     |
| period_end  | M2 (own)                       | iv125     |      11 |     0.117 |       11 |      0.117 | nan     |
| period_end  | SS claims-only path (ml_fixed) | baseline  |      11 |     0.224 |       11 |      0.224 | nan     |
| period_end  | SS claims-only path (ml_fixed) | point     |      11 |     0.144 |       11 |      0.144 | nan     |
| period_end  | SS claims-only path (ml_fixed) | iv110     |      11 |     0.165 |       11 |      0.165 | nan     |
| period_end  | SS claims-only path (ml_fixed) | iv115     |      11 |     0.175 |       11 |      0.175 | nan     |
| period_end  | SS claims-only path (ml_fixed) | iv125     |      11 |     0.192 |       11 |      0.192 | nan     |
| period_end  | N naive last claim             | -         |      13 |     0.329 |       13 |      0.329 | nan     |
| publication | A claims only                  | baseline  |      15 |     0.223 |        8 |      0.358 |   0.6   |
| publication | A claims only                  | point     |      15 |     0.101 |        8 |      0.141 |   0.8   |
| publication | A claims only                  | iv110     |      15 |     0.117 |        8 |      0.172 |   0.8   |
| publication | A claims only                  | iv115     |      15 |     0.124 |        8 |      0.186 |   0.733 |
| publication | A claims only                  | iv125     |      15 |     0.138 |        8 |      0.211 |   0.733 |
| publication | B + indicators                 | baseline  |      15 |     0.202 |        8 |      0.323 |   0.6   |
| publication | B + indicators                 | point     |      15 |     0.123 |        8 |      0.18  |   0.6   |
| publication | B + indicators                 | iv110     |      15 |     0.133 |        8 |      0.199 |   0.6   |
| publication | B + indicators                 | iv115     |      15 |     0.139 |        8 |      0.21  |   0.6   |
| publication | B + indicators                 | iv125     |      15 |     0.149 |        8 |      0.228 |   0.6   |
| publication | E OpenRouter tokens x price    | baseline  |      15 |     0.131 |        8 |      0.189 |   0.8   |
| publication | E OpenRouter tokens x price    | point     |      15 |     0.091 |        8 |      0.117 |   0.933 |
| publication | E OpenRouter tokens x price    | iv110     |      15 |     0.094 |        8 |      0.126 |   0.867 |
| publication | E OpenRouter tokens x price    | iv115     |      15 |     0.097 |        8 |      0.13  |   0.867 |
| publication | E OpenRouter tokens x price    | iv125     |      15 |     0.082 |        8 |      0.104 |   1     |
| publication | G GitHub new-repo share        | baseline  |      15 |     0.1   |        8 |      0.164 |   0.867 |
| publication | G GitHub new-repo share        | point     |      15 |     0.072 |        8 |      0.1   |   0.933 |
| publication | G GitHub new-repo share        | iv110     |      15 |     0.069 |        8 |      0.098 |   0.933 |
| publication | G GitHub new-repo share        | iv115     |      15 |     0.069 |        8 |      0.1   |   0.933 |
| publication | G GitHub new-repo share        | iv125     |      15 |     0.067 |        8 |      0.1   |   0.933 |
| publication | R Ramp adoption                | baseline  |      15 |     0.14  |        8 |      0.22  |   0.867 |
| publication | R Ramp adoption                | point     |      15 |     0.118 |        8 |      0.162 |   0.933 |
| publication | R Ramp adoption                | iv110     |      15 |     0.113 |        8 |      0.154 |   0.933 |
| publication | R Ramp adoption                | iv115     |      15 |     0.113 |        8 |      0.155 |   0.933 |
| publication | R Ramp adoption                | iv125     |      15 |     0.115 |        8 |      0.16  |   0.933 |
| publication | Bayes + indicators             | baseline  |      15 |     0.171 |        8 |      0.276 |   0.8   |
| publication | Bayes + indicators             | iv115     |      15 |     0.117 |        8 |      0.171 |   0.8   |
| publication | M1 (pooled)                    | baseline  |      14 |     0.245 |        7 |      0.42  | nan     |
| publication | M1 (pooled)                    | point     |      14 |     0.269 |        7 |      0.481 | nan     |
| publication | M1 (pooled)                    | iv110     |      14 |     0.263 |        7 |      0.47  | nan     |
| publication | M1 (pooled)                    | iv115     |      14 |     0.26  |        7 |      0.464 | nan     |
| publication | M1 (pooled)                    | iv125     |      14 |     0.256 |        7 |      0.455 | nan     |
| publication | M2 (pooled)                    | baseline  |      14 |     0.118 |        7 |      0.168 | nan     |
| publication | M2 (pooled)                    | point     |      14 |     0.165 |        7 |      0.274 | nan     |
| publication | M2 (pooled)                    | iv110     |      14 |     0.157 |        7 |      0.259 | nan     |
| publication | M2 (pooled)                    | iv115     |      14 |     0.153 |        7 |      0.251 | nan     |
| publication | M2 (pooled)                    | iv125     |      14 |     0.147 |        7 |      0.238 | nan     |
| publication | M1 (own)                       | baseline  |      14 |     0.199 |        7 |      0.328 | nan     |
| publication | M1 (own)                       | point     |      14 |     0.225 |        7 |      0.393 | nan     |
| publication | M1 (own)                       | iv110     |      14 |     0.218 |        7 |      0.38  | nan     |
| publication | M1 (own)                       | iv115     |      14 |     0.215 |        7 |      0.374 | nan     |
| publication | M1 (own)                       | iv125     |      14 |     0.211 |        7 |      0.365 | nan     |
| publication | M2 (own)                       | baseline  |      14 |     0.096 |        7 |      0.123 | nan     |
| publication | M2 (own)                       | point     |      14 |     0.143 |        7 |      0.231 | nan     |
| publication | M2 (own)                       | iv110     |      14 |     0.135 |        7 |      0.215 | nan     |
| publication | M2 (own)                       | iv115     |      14 |     0.131 |        7 |      0.208 | nan     |
| publication | M2 (own)                       | iv125     |      14 |     0.126 |        7 |      0.195 | nan     |
| publication | SS claims-only path (ml_fixed) | baseline  |      14 |     0.207 |        7 |      0.347 | nan     |
| publication | SS claims-only path (ml_fixed) | point     |      14 |     0.107 |        7 |      0.159 | nan     |
| publication | SS claims-only path (ml_fixed) | iv110     |      14 |     0.118 |        7 |      0.183 | nan     |
| publication | SS claims-only path (ml_fixed) | iv115     |      14 |     0.124 |        7 |      0.193 | nan     |
| publication | SS claims-only path (ml_fixed) | iv125     |      14 |     0.134 |        7 |      0.213 | nan     |
| publication | N naive last claim             | -         |      15 |     0.262 |        8 |      0.365 | nan     |

## Variant ranking per model (pre-registered primary: iv115)

| rule        | model                          | on   | order (best first)                                                     | best     |   iv115 rank |
|:------------|:-------------------------------|:-----|:-----------------------------------------------------------------------|:---------|-------------:|
| period_end  | A claims only                  | all  | point 0.144 < iv110 0.165 < iv115 0.175 < iv125 0.192 < baseline 0.224 | point    |            3 |
| period_end  | A claims only                  | post | point 0.144 < iv110 0.165 < iv115 0.175 < iv125 0.192 < baseline 0.224 | point    |            3 |
| period_end  | B + indicators                 | all  | point 0.207 < iv110 0.221 < iv115 0.227 < iv125 0.237 < baseline 0.252 | point    |            3 |
| period_end  | B + indicators                 | post | point 0.207 < iv110 0.221 < iv115 0.227 < iv125 0.237 < baseline 0.252 | point    |            3 |
| period_end  | E OpenRouter tokens x price    | all  | iv115 0.123 < iv110 0.123 < iv125 0.123 < point 0.124 < baseline 0.139 | iv115    |            1 |
| period_end  | E OpenRouter tokens x price    | post | iv115 0.123 < iv110 0.123 < iv125 0.123 < point 0.124 < baseline 0.139 | iv115    |            1 |
| period_end  | G GitHub new-repo share        | all  | iv115 0.107 < iv125 0.108 < iv110 0.108 < point 0.113 < baseline 0.151 | iv115    |            1 |
| period_end  | G GitHub new-repo share        | post | iv115 0.107 < iv125 0.108 < iv110 0.108 < point 0.113 < baseline 0.151 | iv115    |            1 |
| period_end  | R Ramp adoption                | all  | point 0.387 < iv110 0.390 < iv115 0.393 < iv125 0.398 < baseline 0.506 | point    |            3 |
| period_end  | R Ramp adoption                | post | point 0.387 < iv110 0.390 < iv115 0.393 < iv125 0.398 < baseline 0.506 | point    |            3 |
| period_end  | Bayes + indicators             | all  | iv115 0.207 < baseline 0.231                                           | iv115    |            1 |
| period_end  | Bayes + indicators             | post | iv115 0.207 < baseline 0.231                                           | iv115    |            1 |
| period_end  | M1 (pooled)                    | all  | baseline 0.286 < iv125 0.301 < iv115 0.308 < iv110 0.312 < point 0.320 | baseline |            3 |
| period_end  | M1 (pooled)                    | post | baseline 0.286 < iv125 0.301 < iv115 0.308 < iv110 0.312 < point 0.320 | baseline |            3 |
| period_end  | M2 (pooled)                    | all  | baseline 0.122 < iv125 0.144 < iv115 0.154 < iv110 0.160 < point 0.173 | baseline |            3 |
| period_end  | M2 (pooled)                    | post | baseline 0.122 < iv125 0.144 < iv115 0.154 < iv110 0.160 < point 0.173 | baseline |            3 |
| period_end  | M1 (own)                       | all  | baseline 0.232 < iv125 0.248 < iv115 0.254 < iv110 0.258 < point 0.266 | baseline |            3 |
| period_end  | M1 (own)                       | post | baseline 0.232 < iv125 0.248 < iv115 0.254 < iv110 0.258 < point 0.266 | baseline |            3 |
| period_end  | M2 (own)                       | all  | baseline 0.094 < iv125 0.117 < iv115 0.128 < iv110 0.134 < point 0.146 | baseline |            3 |
| period_end  | M2 (own)                       | post | baseline 0.094 < iv125 0.117 < iv115 0.128 < iv110 0.134 < point 0.146 | baseline |            3 |
| period_end  | SS claims-only path (ml_fixed) | all  | point 0.144 < iv110 0.165 < iv115 0.175 < iv125 0.192 < baseline 0.224 | point    |            3 |
| period_end  | SS claims-only path (ml_fixed) | post | point 0.144 < iv110 0.165 < iv115 0.175 < iv125 0.192 < baseline 0.224 | point    |            3 |
| publication | A claims only                  | all  | point 0.107 < iv110 0.118 < iv115 0.124 < iv125 0.134 < baseline 0.207 | point    |            3 |
| publication | A claims only                  | post | point 0.159 < iv110 0.183 < iv115 0.193 < iv125 0.213 < baseline 0.347 | point    |            3 |
| publication | B + indicators                 | all  | point 0.130 < iv110 0.140 < iv115 0.144 < iv125 0.153 < baseline 0.204 | point    |            3 |
| publication | B + indicators                 | post | point 0.203 < iv110 0.222 < iv115 0.231 < iv125 0.246 < baseline 0.345 | point    |            3 |
| publication | E OpenRouter tokens x price    | all  | iv125 0.074 < iv110 0.093 < iv115 0.094 < point 0.096 < baseline 0.104 | iv125    |            3 |
| publication | E OpenRouter tokens x price    | post | iv125 0.090 < iv115 0.130 < iv110 0.130 < point 0.132 < baseline 0.142 | iv125    |            2 |
| publication | G GitHub new-repo share        | all  | iv110 0.063 < point 0.064 < iv115 0.064 < iv125 0.068 < baseline 0.099 | iv110    |            3 |
| publication | G GitHub new-repo share        | post | point 0.086 < iv110 0.091 < iv115 0.095 < iv125 0.106 < baseline 0.171 | point    |            3 |
| publication | R Ramp adoption                | all  | iv110 0.108 < iv115 0.110 < point 0.111 < iv125 0.114 < baseline 0.137 | iv110    |            2 |
| publication | R Ramp adoption                | post | iv110 0.151 < point 0.153 < iv115 0.155 < iv125 0.166 < baseline 0.224 | iv110    |            3 |
| publication | Bayes + indicators             | all  | iv115 0.121 < baseline 0.167                                           | iv115    |            1 |
| publication | Bayes + indicators             | post | iv115 0.187 < baseline 0.283                                           | iv115    |            1 |
| publication | M1 (pooled)                    | all  | baseline 0.245 < iv125 0.256 < iv115 0.260 < iv110 0.263 < point 0.269 | baseline |            3 |
| publication | M1 (pooled)                    | post | baseline 0.420 < iv125 0.455 < iv115 0.464 < iv110 0.470 < point 0.481 | baseline |            3 |
| publication | M2 (pooled)                    | all  | baseline 0.118 < iv125 0.147 < iv115 0.153 < iv110 0.157 < point 0.165 | baseline |            3 |
| publication | M2 (pooled)                    | post | baseline 0.168 < iv125 0.238 < iv115 0.251 < iv110 0.259 < point 0.274 | baseline |            3 |
| publication | M1 (own)                       | all  | baseline 0.199 < iv125 0.211 < iv115 0.215 < iv110 0.218 < point 0.225 | baseline |            3 |
| publication | M1 (own)                       | post | baseline 0.328 < iv125 0.365 < iv115 0.374 < iv110 0.380 < point 0.393 | baseline |            3 |
| publication | M2 (own)                       | all  | baseline 0.096 < iv125 0.126 < iv115 0.131 < iv110 0.135 < point 0.143 | baseline |            3 |
| publication | M2 (own)                       | post | baseline 0.123 < iv125 0.195 < iv115 0.208 < iv110 0.215 < point 0.231 | baseline |            3 |
| publication | SS claims-only path (ml_fixed) | all  | point 0.107 < iv110 0.118 < iv115 0.124 < iv125 0.134 < baseline 0.207 | point    |            3 |
| publication | SS claims-only path (ml_fixed) | post | point 0.159 < iv110 0.183 < iv115 0.193 < iv125 0.213 < baseline 0.347 | point    |            3 |

## Paired differences in |log error| (`B vs A`: negative = B better; 90% bootstrap interval)

Common targets of all models:

| subset   | rule        | comparison                                                                          |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:---------|:------------|:------------------------------------------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| all      | publication | A claims only [iv115] vs A claims only [baseline]                                   |  14 |      -0.083 |    -0.16  |    -0.017 |         12 |       2 |
| all      | publication | A claims only [iv115] vs N naive last claim                                         |  14 |      -0.097 |    -0.16  |    -0.033 |         12 |       2 |
| all      | publication | A claims only [baseline] vs N naive last claim                                      |  14 |      -0.014 |    -0.125 |     0.11  |         11 |       3 |
| all      | publication | B + indicators [iv115] vs B + indicators [baseline]                                 |  14 |      -0.06  |    -0.116 |    -0.01  |         10 |       4 |
| all      | publication | B + indicators [iv115] vs N naive last claim                                        |  14 |      -0.076 |    -0.18  |     0.026 |         11 |       3 |
| all      | publication | B + indicators [baseline] vs N naive last claim                                     |  14 |      -0.017 |    -0.155 |     0.133 |         11 |       3 |
| all      | publication | E OpenRouter tokens x price [iv115] vs E OpenRouter tokens x price [baseline]       |  14 |      -0.009 |    -0.034 |     0.014 |          8 |       6 |
| all      | publication | E OpenRouter tokens x price [iv115] vs N naive last claim                           |  14 |      -0.126 |    -0.22  |    -0.037 |         11 |       3 |
| all      | publication | E OpenRouter tokens x price [baseline] vs N naive last claim                        |  14 |      -0.117 |    -0.21  |    -0.028 |          9 |       5 |
| all      | publication | G GitHub new-repo share [iv115] vs G GitHub new-repo share [baseline]               |  14 |      -0.035 |    -0.073 |    -0.003 |          9 |       5 |
| all      | publication | G GitHub new-repo share [iv115] vs N naive last claim                               |  14 |      -0.157 |    -0.238 |    -0.087 |         13 |       1 |
| all      | publication | G GitHub new-repo share [baseline] vs N naive last claim                            |  14 |      -0.121 |    -0.211 |    -0.034 |         12 |       2 |
| all      | publication | R Ramp adoption [iv115] vs R Ramp adoption [baseline]                               |  14 |      -0.027 |    -0.058 |     0.001 |          7 |       7 |
| all      | publication | R Ramp adoption [iv115] vs N naive last claim                                       |  14 |      -0.111 |    -0.169 |    -0.059 |         13 |       1 |
| all      | publication | R Ramp adoption [baseline] vs N naive last claim                                    |  14 |      -0.084 |    -0.139 |    -0.029 |         12 |       2 |
| all      | publication | Bayes + indicators [iv115] vs Bayes + indicators [baseline]                         |  14 |      -0.046 |    -0.103 |     0.004 |          7 |       7 |
| all      | publication | Bayes + indicators [iv115] vs N naive last claim                                    |  14 |      -0.099 |    -0.201 |    -0.002 |         11 |       3 |
| all      | publication | Bayes + indicators [baseline] vs N naive last claim                                 |  14 |      -0.053 |    -0.192 |     0.09  |         11 |       3 |
| all      | publication | M1 (pooled) [iv115] vs M1 (pooled) [baseline]                                       |  14 |       0.015 |    -0.005 |     0.038 |          8 |       6 |
| all      | publication | M1 (pooled) [iv115] vs N naive last claim                                           |  14 |       0.039 |    -0.129 |     0.236 |         11 |       3 |
| all      | publication | M1 (pooled) [baseline] vs N naive last claim                                        |  14 |       0.024 |    -0.128 |     0.2   |         11 |       3 |
| all      | publication | M2 (pooled) [iv115] vs M2 (pooled) [baseline]                                       |  14 |       0.035 |    -0.003 |     0.078 |          9 |       5 |
| all      | publication | M2 (pooled) [iv115] vs N naive last claim                                           |  14 |      -0.067 |    -0.161 |     0.034 |         11 |       3 |
| all      | publication | M2 (pooled) [baseline] vs N naive last claim                                        |  14 |      -0.102 |    -0.171 |    -0.033 |         12 |       2 |
| all      | publication | M1 (own) [iv115] vs M1 (own) [baseline]                                             |  14 |       0.016 |    -0.004 |     0.038 |          8 |       6 |
| all      | publication | M1 (own) [iv115] vs N naive last claim                                              |  14 |      -0.005 |    -0.156 |     0.16  |         11 |       3 |
| all      | publication | M1 (own) [baseline] vs N naive last claim                                           |  14 |      -0.021 |    -0.157 |     0.125 |         11 |       3 |
| all      | publication | M2 (own) [iv115] vs M2 (own) [baseline]                                             |  14 |       0.036 |    -0.003 |     0.079 |          9 |       5 |
| all      | publication | M2 (own) [iv115] vs N naive last claim                                              |  14 |      -0.089 |    -0.176 |    -0.001 |         11 |       3 |
| all      | publication | M2 (own) [baseline] vs N naive last claim                                           |  14 |      -0.125 |    -0.19  |    -0.066 |         13 |       1 |
| all      | publication | SS claims-only path (ml_fixed) [iv115] vs SS claims-only path (ml_fixed) [baseline] |  14 |      -0.083 |    -0.16  |    -0.017 |         12 |       2 |
| all      | publication | SS claims-only path (ml_fixed) [iv115] vs N naive last claim                        |  14 |      -0.097 |    -0.16  |    -0.033 |         12 |       2 |
| all      | publication | SS claims-only path (ml_fixed) [baseline] vs N naive last claim                     |  14 |      -0.014 |    -0.125 |     0.11  |         11 |       3 |
| post-t0  | publication | A claims only [iv115] vs A claims only [baseline]                                   |   7 |      -0.153 |    -0.288 |    -0.022 |          5 |       2 |
| post-t0  | publication | A claims only [iv115] vs N naive last claim                                         |   7 |      -0.103 |    -0.227 |     0.025 |          5 |       2 |
| post-t0  | publication | A claims only [baseline] vs N naive last claim                                      |   7 |       0.051 |    -0.179 |     0.284 |          4 |       3 |
| post-t0  | publication | B + indicators [iv115] vs B + indicators [baseline]                                 |   7 |      -0.115 |    -0.217 |    -0.019 |          5 |       2 |
| post-t0  | publication | B + indicators [iv115] vs N naive last claim                                        |   7 |      -0.066 |    -0.276 |     0.14  |          4 |       3 |
| post-t0  | publication | B + indicators [baseline] vs N naive last claim                                     |   7 |       0.049 |    -0.237 |     0.328 |          4 |       3 |
| post-t0  | publication | E OpenRouter tokens x price [iv115] vs E OpenRouter tokens x price [baseline]       |   7 |      -0.013 |    -0.045 |     0.016 |          4 |       3 |
| post-t0  | publication | E OpenRouter tokens x price [iv115] vs N naive last claim                           |   7 |      -0.166 |    -0.351 |     0.015 |          4 |       3 |
| post-t0  | publication | E OpenRouter tokens x price [baseline] vs N naive last claim                        |   7 |      -0.154 |    -0.326 |     0.021 |          4 |       3 |
| post-t0  | publication | G GitHub new-repo share [iv115] vs G GitHub new-repo share [baseline]               |   7 |      -0.075 |    -0.136 |    -0.019 |          6 |       1 |
| post-t0  | publication | G GitHub new-repo share [iv115] vs N naive last claim                               |   7 |      -0.201 |    -0.354 |    -0.061 |          6 |       1 |
| post-t0  | publication | G GitHub new-repo share [baseline] vs N naive last claim                            |   7 |      -0.125 |    -0.305 |     0.046 |          5 |       2 |
| post-t0  | publication | R Ramp adoption [iv115] vs R Ramp adoption [baseline]                               |   7 |      -0.069 |    -0.115 |    -0.027 |          6 |       1 |
| post-t0  | publication | R Ramp adoption [iv115] vs N naive last claim                                       |   7 |      -0.141 |    -0.248 |    -0.037 |          6 |       1 |
| post-t0  | publication | R Ramp adoption [baseline] vs N naive last claim                                    |   7 |      -0.072 |    -0.177 |     0.034 |          5 |       2 |
| post-t0  | publication | Bayes + indicators [iv115] vs Bayes + indicators [baseline]                         |   7 |      -0.096 |    -0.201 |     0.002 |          4 |       3 |
| post-t0  | publication | Bayes + indicators [iv115] vs N naive last claim                                    |   7 |      -0.109 |    -0.315 |     0.088 |          4 |       3 |
| post-t0  | publication | Bayes + indicators [baseline] vs N naive last claim                                 |   7 |      -0.013 |    -0.295 |     0.267 |          4 |       3 |
| post-t0  | publication | M1 (pooled) [iv115] vs M1 (pooled) [baseline]                                       |   7 |       0.044 |     0.01  |     0.081 |          1 |       6 |
| post-t0  | publication | M1 (pooled) [iv115] vs N naive last claim                                           |   7 |       0.168 |    -0.172 |     0.542 |          4 |       3 |
| post-t0  | publication | M1 (pooled) [baseline] vs N naive last claim                                        |   7 |       0.123 |    -0.187 |     0.461 |          4 |       3 |
| post-t0  | publication | M2 (pooled) [iv115] vs M2 (pooled) [baseline]                                       |   7 |       0.083 |     0.012 |     0.156 |          2 |       5 |
| post-t0  | publication | M2 (pooled) [iv115] vs N naive last claim                                           |   7 |      -0.045 |    -0.237 |     0.161 |          4 |       3 |
| post-t0  | publication | M2 (pooled) [baseline] vs N naive last claim                                        |   7 |      -0.128 |    -0.264 |     0.014 |          5 |       2 |
| post-t0  | publication | M1 (own) [iv115] vs M1 (own) [baseline]                                             |   7 |       0.046 |     0.012 |     0.081 |          1 |       6 |
| post-t0  | publication | M1 (own) [iv115] vs N naive last claim                                              |   7 |       0.078 |    -0.233 |     0.4   |          4 |       3 |
| post-t0  | publication | M1 (own) [baseline] vs N naive last claim                                           |   7 |       0.032 |    -0.249 |     0.323 |          4 |       3 |
| post-t0  | publication | M2 (own) [iv115] vs M2 (own) [baseline]                                             |   7 |       0.085 |     0.014 |     0.157 |          2 |       5 |
| post-t0  | publication | M2 (own) [iv115] vs N naive last claim                                              |   7 |      -0.089 |    -0.265 |     0.089 |          4 |       3 |
| post-t0  | publication | M2 (own) [baseline] vs N naive last claim                                           |   7 |      -0.173 |    -0.292 |    -0.057 |          6 |       1 |
| post-t0  | publication | SS claims-only path (ml_fixed) [iv115] vs SS claims-only path (ml_fixed) [baseline] |   7 |      -0.153 |    -0.288 |    -0.022 |          5 |       2 |
| post-t0  | publication | SS claims-only path (ml_fixed) [iv115] vs N naive last claim                        |   7 |      -0.103 |    -0.227 |     0.025 |          5 |       2 |
| post-t0  | publication | SS claims-only path (ml_fixed) [baseline] vs N naive last claim                     |   7 |       0.051 |    -0.179 |     0.284 |          4 |       3 |
| H2-2026  | publication | A claims only [iv115] vs A claims only [baseline]                                   |   3 |      -0.355 |    -0.493 |    -0.216 |          3 |       0 |
| H2-2026  | publication | A claims only [iv115] vs N naive last claim                                         |   3 |       0.1   |     0.033 |     0.168 |          1 |       2 |
| H2-2026  | publication | A claims only [baseline] vs N naive last claim                                      |   3 |       0.455 |     0.361 |     0.549 |          0 |       3 |
| H2-2026  | publication | B + indicators [iv115] vs B + indicators [baseline]                                 |   3 |      -0.254 |    -0.369 |    -0.139 |          3 |       0 |
| H2-2026  | publication | B + indicators [iv115] vs N naive last claim                                        |   3 |       0.279 |     0.263 |     0.296 |          0 |       3 |
| H2-2026  | publication | B + indicators [baseline] vs N naive last claim                                     |   3 |       0.533 |     0.42  |     0.647 |          0 |       3 |
| H2-2026  | publication | E OpenRouter tokens x price [iv115] vs E OpenRouter tokens x price [baseline]       |   3 |      -0.016 |    -0.067 |     0.035 |          1 |       2 |
| H2-2026  | publication | E OpenRouter tokens x price [iv115] vs N naive last claim                           |   3 |       0.123 |     0.033 |     0.213 |          0 |       3 |
| H2-2026  | publication | E OpenRouter tokens x price [baseline] vs N naive last claim                        |   3 |       0.138 |     0.055 |     0.222 |          0 |       3 |
| H2-2026  | publication | G GitHub new-repo share [iv115] vs G GitHub new-repo share [baseline]               |   3 |      -0.142 |    -0.228 |    -0.055 |          2 |       1 |
| H2-2026  | publication | G GitHub new-repo share [iv115] vs N naive last claim                               |   3 |       0.004 |    -0.06  |     0.069 |          2 |       1 |
| H2-2026  | publication | G GitHub new-repo share [baseline] vs N naive last claim                            |   3 |       0.146 |     0.041 |     0.251 |          1 |       2 |
| H2-2026  | publication | R Ramp adoption [iv115] vs R Ramp adoption [baseline]                               |   3 |      -0.069 |    -0.119 |    -0.02  |          2 |       1 |
| H2-2026  | publication | R Ramp adoption [iv115] vs N naive last claim                                       |   3 |      -0.007 |    -0.078 |     0.064 |          2 |       1 |
| H2-2026  | publication | R Ramp adoption [baseline] vs N naive last claim                                    |   3 |       0.062 |    -0.019 |     0.144 |          1 |       2 |
| H2-2026  | publication | Bayes + indicators [iv115] vs Bayes + indicators [baseline]                         |   3 |      -0.243 |    -0.357 |    -0.128 |          3 |       0 |
| H2-2026  | publication | Bayes + indicators [iv115] vs N naive last claim                                    |   3 |       0.219 |     0.192 |     0.246 |          0 |       3 |
| H2-2026  | publication | Bayes + indicators [baseline] vs N naive last claim                                 |   3 |       0.462 |     0.362 |     0.562 |          0 |       3 |
| H2-2026  | publication | M1 (pooled) [iv115] vs M1 (pooled) [baseline]                                       |   3 |       0.104 |     0.084 |     0.124 |          0 |       3 |
| H2-2026  | publication | M1 (pooled) [iv115] vs N naive last claim                                           |   3 |       0.751 |     0.498 |     1.004 |          0 |       3 |
| H2-2026  | publication | M1 (pooled) [baseline] vs N naive last claim                                        |   3 |       0.647 |     0.414 |     0.88  |          0 |       3 |
| H2-2026  | publication | M2 (pooled) [iv115] vs M2 (pooled) [baseline]                                       |   3 |       0.199 |     0.144 |     0.254 |          0 |       3 |
| H2-2026  | publication | M2 (pooled) [iv115] vs N naive last claim                                           |   3 |       0.269 |     0.132 |     0.406 |          0 |       3 |
| H2-2026  | publication | M2 (pooled) [baseline] vs N naive last claim                                        |   3 |       0.07  |    -0.049 |     0.189 |          1 |       2 |
| H2-2026  | publication | M1 (own) [iv115] vs M1 (own) [baseline]                                             |   3 |       0.104 |     0.084 |     0.124 |          0 |       3 |
| H2-2026  | publication | M1 (own) [iv115] vs N naive last claim                                              |   3 |       0.607 |     0.427 |     0.788 |          0 |       3 |
| H2-2026  | publication | M1 (own) [baseline] vs N naive last claim                                           |   3 |       0.503 |     0.343 |     0.663 |          0 |       3 |
| H2-2026  | publication | M2 (own) [iv115] vs M2 (own) [baseline]                                             |   3 |       0.2   |     0.144 |     0.255 |          0 |       3 |
| H2-2026  | publication | M2 (own) [iv115] vs N naive last claim                                              |   3 |       0.197 |     0.096 |     0.298 |          0 |       3 |
| H2-2026  | publication | M2 (own) [baseline] vs N naive last claim                                           |   3 |      -0.003 |    -0.068 |     0.062 |          2 |       1 |
| H2-2026  | publication | SS claims-only path (ml_fixed) [iv115] vs SS claims-only path (ml_fixed) [baseline] |   3 |      -0.355 |    -0.493 |    -0.216 |          3 |       0 |
| H2-2026  | publication | SS claims-only path (ml_fixed) [iv115] vs N naive last claim                        |   3 |       0.1   |     0.033 |     0.168 |          1 |       2 |
| H2-2026  | publication | SS claims-only path (ml_fixed) [baseline] vs N naive last claim                     |   3 |       0.455 |     0.361 |     0.549 |          0 |       3 |
| all      | period_end  | A claims only [iv115] vs A claims only [baseline]                                   |  11 |      -0.049 |    -0.091 |    -0.014 |          8 |       3 |
| all      | period_end  | A claims only [iv115] vs N naive last claim                                         |  11 |      -0.082 |    -0.185 |     0.03  |          7 |       4 |
| all      | period_end  | A claims only [baseline] vs N naive last claim                                      |  11 |      -0.033 |    -0.167 |     0.114 |          7 |       4 |
| all      | period_end  | B + indicators [iv115] vs B + indicators [baseline]                                 |  11 |      -0.024 |    -0.042 |    -0.008 |          8 |       3 |
| all      | period_end  | B + indicators [iv115] vs N naive last claim                                        |  11 |      -0.029 |    -0.189 |     0.138 |          7 |       4 |
| all      | period_end  | B + indicators [baseline] vs N naive last claim                                     |  11 |      -0.005 |    -0.178 |     0.179 |          7 |       4 |
| all      | period_end  | E OpenRouter tokens x price [iv115] vs E OpenRouter tokens x price [baseline]       |  11 |      -0.016 |    -0.026 |    -0.006 |         10 |       1 |
| all      | period_end  | E OpenRouter tokens x price [iv115] vs N naive last claim                           |  11 |      -0.133 |    -0.255 |    -0.011 |          7 |       4 |
| all      | period_end  | E OpenRouter tokens x price [baseline] vs N naive last claim                        |  11 |      -0.118 |    -0.238 |     0.006 |          7 |       4 |
| all      | period_end  | G GitHub new-repo share [iv115] vs G GitHub new-repo share [baseline]               |  11 |      -0.044 |    -0.07  |    -0.018 |          9 |       2 |
| all      | period_end  | G GitHub new-repo share [iv115] vs N naive last claim                               |  11 |      -0.15  |    -0.266 |    -0.037 |          9 |       2 |
| all      | period_end  | G GitHub new-repo share [baseline] vs N naive last claim                            |  11 |      -0.106 |    -0.226 |     0.019 |          9 |       2 |
| all      | period_end  | R Ramp adoption [iv115] vs R Ramp adoption [baseline]                               |  11 |      -0.113 |    -0.197 |    -0.037 |          9 |       2 |
| all      | period_end  | R Ramp adoption [iv115] vs N naive last claim                                       |  11 |       0.136 |    -0.104 |     0.413 |          7 |       4 |
| all      | period_end  | R Ramp adoption [baseline] vs N naive last claim                                    |  11 |       0.249 |    -0.04  |     0.584 |          7 |       4 |
| all      | period_end  | Bayes + indicators [iv115] vs Bayes + indicators [baseline]                         |  11 |      -0.023 |    -0.044 |    -0.002 |          9 |       2 |
| all      | period_end  | Bayes + indicators [iv115] vs N naive last claim                                    |  11 |      -0.049 |    -0.207 |     0.112 |          7 |       4 |
| all      | period_end  | Bayes + indicators [baseline] vs N naive last claim                                 |  11 |      -0.026 |    -0.198 |     0.153 |          7 |       4 |
| all      | period_end  | M1 (pooled) [iv115] vs M1 (pooled) [baseline]                                       |  11 |       0.022 |     0.006 |     0.041 |          3 |       8 |
| all      | period_end  | M1 (pooled) [iv115] vs N naive last claim                                           |  11 |       0.051 |    -0.167 |     0.312 |          8 |       3 |
| all      | period_end  | M1 (pooled) [baseline] vs N naive last claim                                        |  11 |       0.029 |    -0.173 |     0.272 |          8 |       3 |
| all      | period_end  | M2 (pooled) [iv115] vs M2 (pooled) [baseline]                                       |  11 |       0.032 |     0.006 |     0.063 |          4 |       7 |
| all      | period_end  | M2 (pooled) [iv115] vs N naive last claim                                           |  11 |      -0.103 |    -0.213 |     0.023 |          8 |       3 |
| all      | period_end  | M2 (pooled) [baseline] vs N naive last claim                                        |  11 |      -0.135 |    -0.226 |    -0.031 |          9 |       2 |
| all      | period_end  | M1 (own) [iv115] vs M1 (own) [baseline]                                             |  11 |       0.022 |     0.006 |     0.04  |          3 |       8 |
| all      | period_end  | M1 (own) [iv115] vs N naive last claim                                              |  11 |      -0.002 |    -0.194 |     0.215 |          8 |       3 |
| all      | period_end  | M1 (own) [baseline] vs N naive last claim                                           |  11 |      -0.024 |    -0.201 |     0.176 |          8 |       3 |
| all      | period_end  | M2 (own) [iv115] vs M2 (own) [baseline]                                             |  11 |       0.034 |     0.008 |     0.064 |          3 |       8 |
| all      | period_end  | M2 (own) [iv115] vs N naive last claim                                              |  11 |      -0.129 |    -0.228 |    -0.023 |          8 |       3 |
| all      | period_end  | M2 (own) [baseline] vs N naive last claim                                           |  11 |      -0.162 |    -0.244 |    -0.08  |         10 |       1 |
| all      | period_end  | SS claims-only path (ml_fixed) [iv115] vs SS claims-only path (ml_fixed) [baseline] |  11 |      -0.049 |    -0.091 |    -0.014 |          8 |       3 |
| all      | period_end  | SS claims-only path (ml_fixed) [iv115] vs N naive last claim                        |  11 |      -0.082 |    -0.185 |     0.03  |          7 |       4 |
| all      | period_end  | SS claims-only path (ml_fixed) [baseline] vs N naive last claim                     |  11 |      -0.033 |    -0.167 |     0.114 |          7 |       4 |
| post-t0  | period_end  | A claims only [iv115] vs A claims only [baseline]                                   |  11 |      -0.049 |    -0.091 |    -0.014 |          8 |       3 |
| post-t0  | period_end  | A claims only [iv115] vs N naive last claim                                         |  11 |      -0.082 |    -0.185 |     0.03  |          7 |       4 |
| post-t0  | period_end  | A claims only [baseline] vs N naive last claim                                      |  11 |      -0.033 |    -0.167 |     0.114 |          7 |       4 |
| post-t0  | period_end  | B + indicators [iv115] vs B + indicators [baseline]                                 |  11 |      -0.024 |    -0.042 |    -0.008 |          8 |       3 |
| post-t0  | period_end  | B + indicators [iv115] vs N naive last claim                                        |  11 |      -0.029 |    -0.189 |     0.138 |          7 |       4 |
| post-t0  | period_end  | B + indicators [baseline] vs N naive last claim                                     |  11 |      -0.005 |    -0.178 |     0.179 |          7 |       4 |
| post-t0  | period_end  | E OpenRouter tokens x price [iv115] vs E OpenRouter tokens x price [baseline]       |  11 |      -0.016 |    -0.026 |    -0.006 |         10 |       1 |
| post-t0  | period_end  | E OpenRouter tokens x price [iv115] vs N naive last claim                           |  11 |      -0.133 |    -0.255 |    -0.011 |          7 |       4 |
| post-t0  | period_end  | E OpenRouter tokens x price [baseline] vs N naive last claim                        |  11 |      -0.118 |    -0.238 |     0.006 |          7 |       4 |
| post-t0  | period_end  | G GitHub new-repo share [iv115] vs G GitHub new-repo share [baseline]               |  11 |      -0.044 |    -0.07  |    -0.018 |          9 |       2 |
| post-t0  | period_end  | G GitHub new-repo share [iv115] vs N naive last claim                               |  11 |      -0.15  |    -0.266 |    -0.037 |          9 |       2 |
| post-t0  | period_end  | G GitHub new-repo share [baseline] vs N naive last claim                            |  11 |      -0.106 |    -0.226 |     0.019 |          9 |       2 |
| post-t0  | period_end  | R Ramp adoption [iv115] vs R Ramp adoption [baseline]                               |  11 |      -0.113 |    -0.197 |    -0.037 |          9 |       2 |
| post-t0  | period_end  | R Ramp adoption [iv115] vs N naive last claim                                       |  11 |       0.136 |    -0.104 |     0.413 |          7 |       4 |
| post-t0  | period_end  | R Ramp adoption [baseline] vs N naive last claim                                    |  11 |       0.249 |    -0.04  |     0.584 |          7 |       4 |
| post-t0  | period_end  | Bayes + indicators [iv115] vs Bayes + indicators [baseline]                         |  11 |      -0.023 |    -0.044 |    -0.002 |          9 |       2 |
| post-t0  | period_end  | Bayes + indicators [iv115] vs N naive last claim                                    |  11 |      -0.049 |    -0.207 |     0.112 |          7 |       4 |
| post-t0  | period_end  | Bayes + indicators [baseline] vs N naive last claim                                 |  11 |      -0.026 |    -0.198 |     0.153 |          7 |       4 |
| post-t0  | period_end  | M1 (pooled) [iv115] vs M1 (pooled) [baseline]                                       |  11 |       0.022 |     0.006 |     0.041 |          3 |       8 |
| post-t0  | period_end  | M1 (pooled) [iv115] vs N naive last claim                                           |  11 |       0.051 |    -0.167 |     0.312 |          8 |       3 |
| post-t0  | period_end  | M1 (pooled) [baseline] vs N naive last claim                                        |  11 |       0.029 |    -0.173 |     0.272 |          8 |       3 |
| post-t0  | period_end  | M2 (pooled) [iv115] vs M2 (pooled) [baseline]                                       |  11 |       0.032 |     0.006 |     0.063 |          4 |       7 |
| post-t0  | period_end  | M2 (pooled) [iv115] vs N naive last claim                                           |  11 |      -0.103 |    -0.213 |     0.023 |          8 |       3 |
| post-t0  | period_end  | M2 (pooled) [baseline] vs N naive last claim                                        |  11 |      -0.135 |    -0.226 |    -0.031 |          9 |       2 |
| post-t0  | period_end  | M1 (own) [iv115] vs M1 (own) [baseline]                                             |  11 |       0.022 |     0.006 |     0.04  |          3 |       8 |
| post-t0  | period_end  | M1 (own) [iv115] vs N naive last claim                                              |  11 |      -0.002 |    -0.194 |     0.215 |          8 |       3 |
| post-t0  | period_end  | M1 (own) [baseline] vs N naive last claim                                           |  11 |      -0.024 |    -0.201 |     0.176 |          8 |       3 |
| post-t0  | period_end  | M2 (own) [iv115] vs M2 (own) [baseline]                                             |  11 |       0.034 |     0.008 |     0.064 |          3 |       8 |
| post-t0  | period_end  | M2 (own) [iv115] vs N naive last claim                                              |  11 |      -0.129 |    -0.228 |    -0.023 |          8 |       3 |
| post-t0  | period_end  | M2 (own) [baseline] vs N naive last claim                                           |  11 |      -0.162 |    -0.244 |    -0.08  |         10 |       1 |
| post-t0  | period_end  | SS claims-only path (ml_fixed) [iv115] vs SS claims-only path (ml_fixed) [baseline] |  11 |      -0.049 |    -0.091 |    -0.014 |          8 |       3 |
| post-t0  | period_end  | SS claims-only path (ml_fixed) [iv115] vs N naive last claim                        |  11 |      -0.082 |    -0.185 |     0.03  |          7 |       4 |
| post-t0  | period_end  | SS claims-only path (ml_fixed) [baseline] vs N naive last claim                     |  11 |      -0.033 |    -0.167 |     0.114 |          7 |       4 |
| H2-2026  | period_end  | A claims only [iv115] vs A claims only [baseline]                                   |   3 |      -0.171 |    -0.2   |    -0.143 |          3 |       0 |
| H2-2026  | period_end  | A claims only [iv115] vs N naive last claim                                         |   3 |       0.224 |     0.171 |     0.277 |          0 |       3 |
| H2-2026  | period_end  | A claims only [baseline] vs N naive last claim                                      |   3 |       0.395 |     0.326 |     0.465 |          0 |       3 |
| H2-2026  | period_end  | B + indicators [iv115] vs B + indicators [baseline]                                 |   3 |      -0.077 |    -0.091 |    -0.064 |          3 |       0 |
| H2-2026  | period_end  | B + indicators [iv115] vs N naive last claim                                        |   3 |       0.428 |     0.328 |     0.527 |          0 |       3 |
| H2-2026  | period_end  | B + indicators [baseline] vs N naive last claim                                     |   3 |       0.505 |     0.392 |     0.617 |          0 |       3 |
| H2-2026  | period_end  | E OpenRouter tokens x price [iv115] vs E OpenRouter tokens x price [baseline]       |   3 |      -0.029 |    -0.045 |    -0.013 |          3 |       0 |
| H2-2026  | period_end  | E OpenRouter tokens x price [iv115] vs N naive last claim                           |   3 |       0.176 |     0.132 |     0.22  |          0 |       3 |
| H2-2026  | period_end  | E OpenRouter tokens x price [baseline] vs N naive last claim                        |   3 |       0.205 |     0.145 |     0.265 |          0 |       3 |
| H2-2026  | period_end  | G GitHub new-repo share [iv115] vs G GitHub new-repo share [baseline]               |   3 |       0     |    -0.036 |     0.036 |          1 |       2 |
| H2-2026  | period_end  | G GitHub new-repo share [iv115] vs N naive last claim                               |   3 |      -0.009 |    -0.057 |     0.038 |          2 |       1 |
| H2-2026  | period_end  | G GitHub new-repo share [baseline] vs N naive last claim                            |   3 |      -0.009 |    -0.093 |     0.074 |          2 |       1 |
| H2-2026  | period_end  | R Ramp adoption [iv115] vs R Ramp adoption [baseline]                               |   3 |      -0.002 |    -0.03  |     0.027 |          2 |       1 |
| H2-2026  | period_end  | R Ramp adoption [iv115] vs N naive last claim                                       |   3 |      -0.012 |    -0.064 |     0.04  |          2 |       1 |
| H2-2026  | period_end  | R Ramp adoption [baseline] vs N naive last claim                                    |   3 |      -0.011 |    -0.091 |     0.07  |          2 |       1 |
| H2-2026  | period_end  | Bayes + indicators [iv115] vs Bayes + indicators [baseline]                         |   3 |      -0.07  |    -0.078 |    -0.062 |          3 |       0 |
| H2-2026  | period_end  | Bayes + indicators [iv115] vs N naive last claim                                    |   3 |       0.383 |     0.318 |     0.448 |          0 |       3 |
| H2-2026  | period_end  | Bayes + indicators [baseline] vs N naive last claim                                 |   3 |       0.453 |     0.381 |     0.525 |          0 |       3 |
| H2-2026  | period_end  | M1 (pooled) [iv115] vs M1 (pooled) [baseline]                                       |   3 |       0.078 |     0.071 |     0.086 |          0 |       3 |
| H2-2026  | period_end  | M1 (pooled) [iv115] vs N naive last claim                                           |   3 |       0.748 |     0.497 |     0.998 |          0 |       3 |
| H2-2026  | period_end  | M1 (pooled) [baseline] vs N naive last claim                                        |   3 |       0.669 |     0.426 |     0.913 |          0 |       3 |
| H2-2026  | period_end  | M2 (pooled) [iv115] vs M2 (pooled) [baseline]                                       |   3 |       0.125 |     0.107 |     0.143 |          0 |       3 |
| H2-2026  | period_end  | M2 (pooled) [iv115] vs N naive last claim                                           |   3 |       0.205 |     0.064 |     0.347 |          0 |       3 |
| H2-2026  | period_end  | M2 (pooled) [baseline] vs N naive last claim                                        |   3 |       0.081 |    -0.066 |     0.228 |          1 |       2 |
| H2-2026  | period_end  | M1 (own) [iv115] vs M1 (own) [baseline]                                             |   3 |       0.078 |     0.071 |     0.085 |          0 |       3 |
| H2-2026  | period_end  | M1 (own) [iv115] vs N naive last claim                                              |   3 |       0.595 |     0.421 |     0.769 |          0 |       3 |
| H2-2026  | period_end  | M1 (own) [baseline] vs N naive last claim                                           |   3 |       0.517 |     0.35  |     0.684 |          0 |       3 |
| H2-2026  | period_end  | M2 (own) [iv115] vs M2 (own) [baseline]                                             |   3 |       0.125 |     0.107 |     0.143 |          0 |       3 |
| H2-2026  | period_end  | M2 (own) [iv115] vs N naive last claim                                              |   3 |       0.129 |     0.046 |     0.213 |          0 |       3 |
| H2-2026  | period_end  | M2 (own) [baseline] vs N naive last claim                                           |   3 |       0.004 |    -0.085 |     0.094 |          2 |       1 |
| H2-2026  | period_end  | SS claims-only path (ml_fixed) [iv115] vs SS claims-only path (ml_fixed) [baseline] |   3 |      -0.171 |    -0.2   |    -0.143 |          3 |       0 |
| H2-2026  | period_end  | SS claims-only path (ml_fixed) [iv115] vs N naive last claim                        |   3 |       0.224 |     0.171 |     0.277 |          0 |       3 |
| H2-2026  | period_end  | SS claims-only path (ml_fixed) [baseline] vs N naive last claim                     |   3 |       0.395 |     0.326 |     0.465 |          0 |       3 |

A, E, G, R on all their targets, every variant vs baseline:

| subset   | rule        | comparison                                                                    |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:---------|:------------|:------------------------------------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| all      | publication | A claims only [iv115] vs A claims only [baseline]                             |  15 |      -0.098 |    -0.173 |    -0.034 |         13 |       2 |
| all      | publication | A claims only [iv115] vs N naive last claim                                   |  15 |      -0.138 |    -0.23  |    -0.054 |         13 |       2 |
| all      | publication | A claims only [baseline] vs N naive last claim                                |  15 |      -0.04  |    -0.151 |     0.083 |         12 |       3 |
| all      | publication | A claims only [point] vs A claims only [baseline]                             |  15 |      -0.122 |    -0.211 |    -0.043 |         12 |       3 |
| all      | publication | A claims only [iv110] vs A claims only [baseline]                             |  15 |      -0.106 |    -0.184 |    -0.037 |         13 |       2 |
| all      | publication | A claims only [iv125] vs A claims only [baseline]                             |  15 |      -0.084 |    -0.15  |    -0.027 |         13 |       2 |
| all      | publication | E OpenRouter tokens x price [iv115] vs E OpenRouter tokens x price [baseline] |  15 |      -0.034 |    -0.084 |     0.006 |          9 |       6 |
| all      | publication | E OpenRouter tokens x price [iv115] vs N naive last claim                     |  15 |      -0.166 |    -0.273 |    -0.062 |         12 |       3 |
| all      | publication | E OpenRouter tokens x price [baseline] vs N naive last claim                  |  15 |      -0.131 |    -0.22  |    -0.044 |         10 |       5 |
| all      | publication | E OpenRouter tokens x price [point] vs E OpenRouter tokens x price [baseline] |  15 |      -0.04  |    -0.105 |     0.012 |          8 |       7 |
| all      | publication | E OpenRouter tokens x price [iv110] vs E OpenRouter tokens x price [baseline] |  15 |      -0.037 |    -0.091 |     0.007 |          9 |       6 |
| all      | publication | E OpenRouter tokens x price [iv125] vs E OpenRouter tokens x price [baseline] |  15 |      -0.049 |    -0.094 |    -0.009 |         10 |       5 |
| all      | publication | G GitHub new-repo share [iv115] vs G GitHub new-repo share [baseline]         |  15 |      -0.031 |    -0.068 |    -0.001 |          9 |       6 |
| all      | publication | G GitHub new-repo share [iv115] vs N naive last claim                         |  15 |      -0.194 |    -0.291 |    -0.106 |         14 |       1 |
| all      | publication | G GitHub new-repo share [baseline] vs N naive last claim                      |  15 |      -0.162 |    -0.269 |    -0.059 |         13 |       2 |
| all      | publication | G GitHub new-repo share [point] vs G GitHub new-repo share [baseline]         |  15 |      -0.028 |    -0.073 |     0.012 |          9 |       6 |
| all      | publication | G GitHub new-repo share [iv110] vs G GitHub new-repo share [baseline]         |  15 |      -0.031 |    -0.071 |     0.003 |          9 |       6 |
| all      | publication | G GitHub new-repo share [iv125] vs G GitHub new-repo share [baseline]         |  15 |      -0.033 |    -0.062 |    -0.007 |         11 |       4 |
| all      | publication | R Ramp adoption [iv115] vs R Ramp adoption [baseline]                         |  15 |      -0.028 |    -0.057 |    -0.001 |          8 |       7 |
| all      | publication | R Ramp adoption [iv115] vs N naive last claim                                 |  15 |      -0.15  |    -0.237 |    -0.074 |         14 |       1 |
| all      | publication | R Ramp adoption [baseline] vs N naive last claim                              |  15 |      -0.122 |    -0.207 |    -0.046 |         13 |       2 |
| all      | publication | R Ramp adoption [point] vs R Ramp adoption [baseline]                         |  15 |      -0.022 |    -0.057 |     0.008 |          7 |       8 |
| all      | publication | R Ramp adoption [iv110] vs R Ramp adoption [baseline]                         |  15 |      -0.028 |    -0.059 |     0.001 |          8 |       7 |
| all      | publication | R Ramp adoption [iv125] vs R Ramp adoption [baseline]                         |  15 |      -0.026 |    -0.052 |    -0.002 |          8 |       7 |
| post-t0  | publication | A claims only [iv115] vs A claims only [baseline]                             |   8 |      -0.173 |    -0.293 |    -0.061 |          6 |       2 |
| post-t0  | publication | A claims only [iv115] vs N naive last claim                                   |   8 |      -0.179 |    -0.344 |    -0.024 |          6 |       2 |
| post-t0  | publication | A claims only [baseline] vs N naive last claim                                |   8 |      -0.007 |    -0.218 |     0.218 |          5 |       3 |
| post-t0  | publication | A claims only [point] vs A claims only [baseline]                             |   8 |      -0.217 |    -0.355 |    -0.084 |          6 |       2 |
| post-t0  | publication | A claims only [iv110] vs A claims only [baseline]                             |   8 |      -0.186 |    -0.312 |    -0.068 |          6 |       2 |
| post-t0  | publication | A claims only [iv125] vs A claims only [baseline]                             |   8 |      -0.148 |    -0.257 |    -0.046 |          6 |       2 |
| post-t0  | publication | E OpenRouter tokens x price [iv115] vs E OpenRouter tokens x price [baseline] |   8 |      -0.059 |    -0.145 |     0.007 |          5 |       3 |
| post-t0  | publication | E OpenRouter tokens x price [iv115] vs N naive last claim                     |   8 |      -0.235 |    -0.423 |    -0.046 |          5 |       3 |
| post-t0  | publication | E OpenRouter tokens x price [baseline] vs N naive last claim                  |   8 |      -0.176 |    -0.325 |    -0.017 |          5 |       3 |
| post-t0  | publication | E OpenRouter tokens x price [point] vs E OpenRouter tokens x price [baseline] |   8 |      -0.072 |    -0.187 |     0.019 |          4 |       4 |
| post-t0  | publication | E OpenRouter tokens x price [iv110] vs E OpenRouter tokens x price [baseline] |   8 |      -0.063 |    -0.156 |     0.01  |          5 |       3 |
| post-t0  | publication | E OpenRouter tokens x price [iv125] vs E OpenRouter tokens x price [baseline] |   8 |      -0.085 |    -0.154 |    -0.025 |          6 |       2 |
| post-t0  | publication | G GitHub new-repo share [iv115] vs G GitHub new-repo share [baseline]         |   8 |      -0.064 |    -0.121 |    -0.012 |          6 |       2 |
| post-t0  | publication | G GitHub new-repo share [iv115] vs N naive last claim                         |   8 |      -0.265 |    -0.429 |    -0.111 |          7 |       1 |
| post-t0  | publication | G GitHub new-repo share [baseline] vs N naive last claim                      |   8 |      -0.201 |    -0.397 |    -0.012 |          6 |       2 |
| post-t0  | publication | G GitHub new-repo share [point] vs G GitHub new-repo share [baseline]         |   8 |      -0.064 |    -0.138 |     0.005 |          6 |       2 |
| post-t0  | publication | G GitHub new-repo share [iv110] vs G GitHub new-repo share [baseline]         |   8 |      -0.065 |    -0.13  |    -0.007 |          6 |       2 |
| post-t0  | publication | G GitHub new-repo share [iv125] vs G GitHub new-repo share [baseline]         |   8 |      -0.064 |    -0.108 |    -0.025 |          8 |       0 |
| post-t0  | publication | R Ramp adoption [iv115] vs R Ramp adoption [baseline]                         |   8 |      -0.065 |    -0.105 |    -0.026 |          7 |       1 |
| post-t0  | publication | R Ramp adoption [iv115] vs N naive last claim                                 |   8 |      -0.21  |    -0.359 |    -0.076 |          7 |       1 |
| post-t0  | publication | R Ramp adoption [baseline] vs N naive last claim                              |   8 |      -0.145 |    -0.296 |    -0.01  |          6 |       2 |
| post-t0  | publication | R Ramp adoption [point] vs R Ramp adoption [baseline]                         |   8 |      -0.058 |    -0.107 |    -0.01  |          6 |       2 |
| post-t0  | publication | R Ramp adoption [iv110] vs R Ramp adoption [baseline]                         |   8 |      -0.066 |    -0.11  |    -0.024 |          7 |       1 |
| post-t0  | publication | R Ramp adoption [iv125] vs R Ramp adoption [baseline]                         |   8 |      -0.06  |    -0.096 |    -0.028 |          7 |       1 |
| H2-2026  | publication | A claims only [iv115] vs A claims only [baseline]                             |   3 |      -0.355 |    -0.493 |    -0.216 |          3 |       0 |
| H2-2026  | publication | A claims only [iv115] vs N naive last claim                                   |   3 |       0.1   |     0.033 |     0.168 |          1 |       2 |
| H2-2026  | publication | A claims only [baseline] vs N naive last claim                                |   3 |       0.455 |     0.361 |     0.549 |          0 |       3 |
| H2-2026  | publication | A claims only [point] vs A claims only [baseline]                             |   3 |      -0.427 |    -0.568 |    -0.286 |          3 |       0 |
| H2-2026  | publication | A claims only [iv110] vs A claims only [baseline]                             |   3 |      -0.378 |    -0.517 |    -0.239 |          3 |       0 |
| H2-2026  | publication | A claims only [iv125] vs A claims only [baseline]                             |   3 |      -0.313 |    -0.449 |    -0.177 |          3 |       0 |
| H2-2026  | publication | E OpenRouter tokens x price [iv115] vs E OpenRouter tokens x price [baseline] |   3 |      -0.016 |    -0.067 |     0.035 |          1 |       2 |
| H2-2026  | publication | E OpenRouter tokens x price [iv115] vs N naive last claim                     |   3 |       0.123 |     0.033 |     0.213 |          0 |       3 |
| H2-2026  | publication | E OpenRouter tokens x price [baseline] vs N naive last claim                  |   3 |       0.138 |     0.055 |     0.222 |          0 |       3 |
| H2-2026  | publication | E OpenRouter tokens x price [point] vs E OpenRouter tokens x price [baseline] |   3 |      -0.01  |    -0.089 |     0.07  |          1 |       2 |
| H2-2026  | publication | E OpenRouter tokens x price [iv110] vs E OpenRouter tokens x price [baseline] |   3 |      -0.014 |    -0.074 |     0.046 |          1 |       2 |
| H2-2026  | publication | E OpenRouter tokens x price [iv125] vs E OpenRouter tokens x price [baseline] |   3 |      -0.111 |    -0.177 |    -0.044 |          2 |       1 |
| H2-2026  | publication | G GitHub new-repo share [iv115] vs G GitHub new-repo share [baseline]         |   3 |      -0.142 |    -0.228 |    -0.055 |          2 |       1 |
| H2-2026  | publication | G GitHub new-repo share [iv115] vs N naive last claim                         |   3 |       0.004 |    -0.06  |     0.069 |          2 |       1 |
| H2-2026  | publication | G GitHub new-repo share [baseline] vs N naive last claim                      |   3 |       0.146 |     0.041 |     0.251 |          1 |       2 |
| H2-2026  | publication | G GitHub new-repo share [point] vs G GitHub new-repo share [baseline]         |   3 |      -0.16  |    -0.269 |    -0.051 |          2 |       1 |
| H2-2026  | publication | G GitHub new-repo share [iv110] vs G GitHub new-repo share [baseline]         |   3 |      -0.152 |    -0.25  |    -0.053 |          2 |       1 |
| H2-2026  | publication | G GitHub new-repo share [iv125] vs G GitHub new-repo share [baseline]         |   3 |      -0.122 |    -0.191 |    -0.054 |          3 |       0 |
| H2-2026  | publication | R Ramp adoption [iv115] vs R Ramp adoption [baseline]                         |   3 |      -0.069 |    -0.119 |    -0.02  |          2 |       1 |
| H2-2026  | publication | R Ramp adoption [iv115] vs N naive last claim                                 |   3 |      -0.007 |    -0.078 |     0.064 |          2 |       1 |
| H2-2026  | publication | R Ramp adoption [baseline] vs N naive last claim                              |   3 |       0.062 |    -0.019 |     0.144 |          1 |       2 |
| H2-2026  | publication | R Ramp adoption [point] vs R Ramp adoption [baseline]                         |   3 |      -0.074 |    -0.143 |    -0.004 |          2 |       1 |
| H2-2026  | publication | R Ramp adoption [iv110] vs R Ramp adoption [baseline]                         |   3 |      -0.079 |    -0.137 |    -0.021 |          2 |       1 |
| H2-2026  | publication | R Ramp adoption [iv125] vs R Ramp adoption [baseline]                         |   3 |      -0.053 |    -0.087 |    -0.019 |          2 |       1 |
| all      | period_end  | A claims only [iv115] vs A claims only [baseline]                             |  13 |      -0.083 |    -0.134 |    -0.035 |         10 |       3 |
| all      | period_end  | A claims only [iv115] vs N naive last claim                                   |  13 |      -0.164 |    -0.291 |    -0.038 |          9 |       4 |
| all      | period_end  | A claims only [baseline] vs N naive last claim                                |  13 |      -0.081 |    -0.206 |     0.051 |          9 |       4 |
| all      | period_end  | A claims only [point] vs A claims only [baseline]                             |  13 |      -0.124 |    -0.198 |    -0.055 |         10 |       3 |
| all      | period_end  | A claims only [iv110] vs A claims only [baseline]                             |  13 |      -0.095 |    -0.153 |    -0.041 |         10 |       3 |
| all      | period_end  | A claims only [iv125] vs A claims only [baseline]                             |  13 |      -0.06  |    -0.099 |    -0.024 |         11 |       2 |
| all      | period_end  | E OpenRouter tokens x price [iv115] vs E OpenRouter tokens x price [baseline] |  13 |      -0.065 |    -0.123 |    -0.014 |         12 |       1 |
| all      | period_end  | E OpenRouter tokens x price [iv115] vs N naive last claim                     |  13 |      -0.207 |    -0.34  |    -0.078 |          9 |       4 |
| all      | period_end  | E OpenRouter tokens x price [baseline] vs N naive last claim                  |  13 |      -0.142 |    -0.248 |    -0.036 |          9 |       4 |
| all      | period_end  | E OpenRouter tokens x price [point] vs E OpenRouter tokens x price [baseline] |  13 |      -0.08  |    -0.158 |    -0.012 |          9 |       4 |
| all      | period_end  | E OpenRouter tokens x price [iv110] vs E OpenRouter tokens x price [baseline] |  13 |      -0.069 |    -0.134 |    -0.014 |         11 |       2 |
| all      | period_end  | E OpenRouter tokens x price [iv125] vs E OpenRouter tokens x price [baseline] |  13 |      -0.056 |    -0.104 |    -0.015 |         11 |       2 |
| all      | period_end  | G GitHub new-repo share [iv115] vs G GitHub new-repo share [baseline]         |  13 |      -0.051 |    -0.083 |    -0.022 |         10 |       3 |
| all      | period_end  | G GitHub new-repo share [iv115] vs N naive last claim                         |  13 |      -0.224 |    -0.353 |    -0.098 |         11 |       2 |
| all      | period_end  | G GitHub new-repo share [baseline] vs N naive last claim                      |  13 |      -0.173 |    -0.306 |    -0.04  |         11 |       2 |
| all      | period_end  | G GitHub new-repo share [point] vs G GitHub new-repo share [baseline]         |  13 |      -0.033 |    -0.068 |     0.003 |         10 |       3 |
| all      | period_end  | G GitHub new-repo share [iv110] vs G GitHub new-repo share [baseline]         |  13 |      -0.05  |    -0.087 |    -0.017 |         10 |       3 |
| all      | period_end  | G GitHub new-repo share [iv125] vs G GitHub new-repo share [baseline]         |  13 |      -0.053 |    -0.078 |    -0.031 |         12 |       1 |
| all      | period_end  | R Ramp adoption [iv115] vs R Ramp adoption [baseline]                         |  13 |      -0.103 |    -0.175 |    -0.04  |         11 |       2 |
| all      | period_end  | R Ramp adoption [iv115] vs N naive last claim                                 |  13 |       0.092 |    -0.138 |     0.347 |          8 |       5 |
| all      | period_end  | R Ramp adoption [baseline] vs N naive last claim                              |  13 |       0.195 |    -0.074 |     0.5   |          8 |       5 |
| all      | period_end  | R Ramp adoption [point] vs R Ramp adoption [baseline]                         |  13 |      -0.104 |    -0.186 |    -0.032 |          9 |       4 |
| all      | period_end  | R Ramp adoption [iv110] vs R Ramp adoption [baseline]                         |  13 |      -0.104 |    -0.179 |    -0.039 |         10 |       3 |
| all      | period_end  | R Ramp adoption [iv125] vs R Ramp adoption [baseline]                         |  13 |      -0.101 |    -0.168 |    -0.043 |         11 |       2 |
| post-t0  | period_end  | A claims only [iv115] vs A claims only [baseline]                             |  13 |      -0.083 |    -0.134 |    -0.035 |         10 |       3 |
| post-t0  | period_end  | A claims only [iv115] vs N naive last claim                                   |  13 |      -0.164 |    -0.291 |    -0.038 |          9 |       4 |
| post-t0  | period_end  | A claims only [baseline] vs N naive last claim                                |  13 |      -0.081 |    -0.206 |     0.051 |          9 |       4 |
| post-t0  | period_end  | A claims only [point] vs A claims only [baseline]                             |  13 |      -0.124 |    -0.198 |    -0.055 |         10 |       3 |
| post-t0  | period_end  | A claims only [iv110] vs A claims only [baseline]                             |  13 |      -0.095 |    -0.153 |    -0.041 |         10 |       3 |
| post-t0  | period_end  | A claims only [iv125] vs A claims only [baseline]                             |  13 |      -0.06  |    -0.099 |    -0.024 |         11 |       2 |
| post-t0  | period_end  | E OpenRouter tokens x price [iv115] vs E OpenRouter tokens x price [baseline] |  13 |      -0.065 |    -0.123 |    -0.014 |         12 |       1 |
| post-t0  | period_end  | E OpenRouter tokens x price [iv115] vs N naive last claim                     |  13 |      -0.207 |    -0.34  |    -0.078 |          9 |       4 |
| post-t0  | period_end  | E OpenRouter tokens x price [baseline] vs N naive last claim                  |  13 |      -0.142 |    -0.248 |    -0.036 |          9 |       4 |
| post-t0  | period_end  | E OpenRouter tokens x price [point] vs E OpenRouter tokens x price [baseline] |  13 |      -0.08  |    -0.158 |    -0.012 |          9 |       4 |
| post-t0  | period_end  | E OpenRouter tokens x price [iv110] vs E OpenRouter tokens x price [baseline] |  13 |      -0.069 |    -0.134 |    -0.014 |         11 |       2 |
| post-t0  | period_end  | E OpenRouter tokens x price [iv125] vs E OpenRouter tokens x price [baseline] |  13 |      -0.056 |    -0.104 |    -0.015 |         11 |       2 |
| post-t0  | period_end  | G GitHub new-repo share [iv115] vs G GitHub new-repo share [baseline]         |  13 |      -0.051 |    -0.083 |    -0.022 |         10 |       3 |
| post-t0  | period_end  | G GitHub new-repo share [iv115] vs N naive last claim                         |  13 |      -0.224 |    -0.353 |    -0.098 |         11 |       2 |
| post-t0  | period_end  | G GitHub new-repo share [baseline] vs N naive last claim                      |  13 |      -0.173 |    -0.306 |    -0.04  |         11 |       2 |
| post-t0  | period_end  | G GitHub new-repo share [point] vs G GitHub new-repo share [baseline]         |  13 |      -0.033 |    -0.068 |     0.003 |         10 |       3 |
| post-t0  | period_end  | G GitHub new-repo share [iv110] vs G GitHub new-repo share [baseline]         |  13 |      -0.05  |    -0.087 |    -0.017 |         10 |       3 |
| post-t0  | period_end  | G GitHub new-repo share [iv125] vs G GitHub new-repo share [baseline]         |  13 |      -0.053 |    -0.078 |    -0.031 |         12 |       1 |
| post-t0  | period_end  | R Ramp adoption [iv115] vs R Ramp adoption [baseline]                         |  13 |      -0.103 |    -0.175 |    -0.04  |         11 |       2 |
| post-t0  | period_end  | R Ramp adoption [iv115] vs N naive last claim                                 |  13 |       0.092 |    -0.138 |     0.347 |          8 |       5 |
| post-t0  | period_end  | R Ramp adoption [baseline] vs N naive last claim                              |  13 |       0.195 |    -0.074 |     0.5   |          8 |       5 |
| post-t0  | period_end  | R Ramp adoption [point] vs R Ramp adoption [baseline]                         |  13 |      -0.104 |    -0.186 |    -0.032 |          9 |       4 |
| post-t0  | period_end  | R Ramp adoption [iv110] vs R Ramp adoption [baseline]                         |  13 |      -0.104 |    -0.179 |    -0.039 |         10 |       3 |
| post-t0  | period_end  | R Ramp adoption [iv125] vs R Ramp adoption [baseline]                         |  13 |      -0.101 |    -0.168 |    -0.043 |         11 |       2 |
| H2-2026  | period_end  | A claims only [iv115] vs A claims only [baseline]                             |   3 |      -0.171 |    -0.2   |    -0.143 |          3 |       0 |
| H2-2026  | period_end  | A claims only [iv115] vs N naive last claim                                   |   3 |       0.224 |     0.171 |     0.277 |          0 |       3 |
| H2-2026  | period_end  | A claims only [baseline] vs N naive last claim                                |   3 |       0.395 |     0.326 |     0.465 |          0 |       3 |
| H2-2026  | period_end  | A claims only [point] vs A claims only [baseline]                             |   3 |      -0.269 |    -0.316 |    -0.223 |          3 |       0 |
| H2-2026  | period_end  | A claims only [iv110] vs A claims only [baseline]                             |   3 |      -0.203 |    -0.237 |    -0.168 |          3 |       0 |
| H2-2026  | period_end  | A claims only [iv125] vs A claims only [baseline]                             |   3 |      -0.117 |    -0.137 |    -0.098 |          3 |       0 |
| H2-2026  | period_end  | E OpenRouter tokens x price [iv115] vs E OpenRouter tokens x price [baseline] |   3 |      -0.029 |    -0.045 |    -0.013 |          3 |       0 |
| H2-2026  | period_end  | E OpenRouter tokens x price [iv115] vs N naive last claim                     |   3 |       0.176 |     0.132 |     0.22  |          0 |       3 |
| H2-2026  | period_end  | E OpenRouter tokens x price [baseline] vs N naive last claim                  |   3 |       0.205 |     0.145 |     0.265 |          0 |       3 |
| H2-2026  | period_end  | E OpenRouter tokens x price [point] vs E OpenRouter tokens x price [baseline] |   3 |      -0.021 |    -0.072 |     0.031 |          1 |       2 |
| H2-2026  | period_end  | E OpenRouter tokens x price [iv110] vs E OpenRouter tokens x price [baseline] |   3 |      -0.027 |    -0.054 |    -0.001 |          2 |       1 |
| H2-2026  | period_end  | E OpenRouter tokens x price [iv125] vs E OpenRouter tokens x price [baseline] |   3 |      -0.031 |    -0.037 |    -0.026 |          3 |       0 |
| H2-2026  | period_end  | G GitHub new-repo share [iv115] vs G GitHub new-repo share [baseline]         |   3 |       0     |    -0.036 |     0.036 |          1 |       2 |
| H2-2026  | period_end  | G GitHub new-repo share [iv115] vs N naive last claim                         |   3 |      -0.009 |    -0.057 |     0.038 |          2 |       1 |
| H2-2026  | period_end  | G GitHub new-repo share [baseline] vs N naive last claim                      |   3 |      -0.009 |    -0.093 |     0.074 |          2 |       1 |
| H2-2026  | period_end  | G GitHub new-repo share [point] vs G GitHub new-repo share [baseline]         |   3 |       0.022 |    -0.059 |     0.103 |          1 |       2 |
| H2-2026  | period_end  | G GitHub new-repo share [iv110] vs G GitHub new-repo share [baseline]         |   3 |       0.005 |    -0.046 |     0.056 |          1 |       2 |
| H2-2026  | period_end  | G GitHub new-repo share [iv125] vs G GitHub new-repo share [baseline]         |   3 |      -0.006 |    -0.022 |     0.009 |          2 |       1 |
| H2-2026  | period_end  | R Ramp adoption [iv115] vs R Ramp adoption [baseline]                         |   3 |      -0.002 |    -0.03  |     0.027 |          2 |       1 |
| H2-2026  | period_end  | R Ramp adoption [iv115] vs N naive last claim                                 |   3 |      -0.012 |    -0.064 |     0.04  |          2 |       1 |
| H2-2026  | period_end  | R Ramp adoption [baseline] vs N naive last claim                              |   3 |      -0.011 |    -0.091 |     0.07  |          2 |       1 |
| H2-2026  | period_end  | R Ramp adoption [point] vs R Ramp adoption [baseline]                         |   3 |       0.008 |    -0.054 |     0.069 |          1 |       2 |
| H2-2026  | period_end  | R Ramp adoption [iv110] vs R Ramp adoption [baseline]                         |   3 |       0.001 |    -0.037 |     0.039 |          1 |       2 |
| H2-2026  | period_end  | R Ramp adoption [iv125] vs R Ramp adoption [baseline]                         |   3 |      -0.004 |    -0.018 |     0.011 |          2 |       1 |

Per-target log error, publication:

|                          |   A baseline |   A iv115 |   A point |   E baseline |   E iv115 |   E point |   G baseline |   G iv115 |   G point |   R baseline |   R iv115 |   R point |   naive |
|:-------------------------|-------------:|----------:|----------:|-------------:|----------:|----------:|-------------:|----------:|----------:|-------------:|----------:|----------:|--------:|
| ('rev_2025fy', False)    |        0.062 |     0.045 |     0.047 |        0.018 |     0.078 |     0.078 |       -0.022 |    -0.013 |    -0.012 |        0.07  |     0.071 |     0.072 |   0.118 |
| ('rev_2025q1', False)    |        0.082 |     0.059 |     0.042 |        0.166 |     0.104 |     0.084 |       -0.023 |    -0.031 |    -0.04  |       -0.063 |    -0.071 |    -0.078 |   0.145 |
| ('rev_2025q2', False)    |        0.067 |     0.05  |     0.055 |        0.032 |     0.077 |     0.08  |       -0.012 |     0.006 |     0.012 |        0.066 |     0.073 |     0.079 |   0.149 |
| ('rev_2025q3', False)    |        0.043 |     0.032 |     0.058 |       -0.17  |     0.05  |     0.081 |       -0.042 |     0.008 |     0.028 |        0.053 |     0.076 |     0.092 |   0.158 |
| ('rev_2026q1', False)    |        0.05  |     0.03  |     0.016 |        0.049 |     0.033 |     0.018 |       -0.05  |    -0.059 |    -0.072 |        0.063 |     0.052 |     0.043 |   0.138 |
| ('rev_2026q2', False)    |       -0.004 |    -0.001 |     0     |       -0.008 |    -0.005 |    -0.003 |        0.005 |     0.005 |     0.006 |        0.006 |     0.007 |     0.008 |   0.056 |
| ('rev_2026q3', True)     |       -0.324 |    -0.198 |    -0.129 |        0.04  |     0.062 |     0.101 |        0.014 |     0.022 |     0.062 |       -0     |     0.021 |     0.049 |  -0.038 |
| ('rr_2025_10', True)     |       -0.44  |    -0.131 |    -0.012 |       -0.517 |    -0.133 |    -0.014 |       -0.113 |     0.132 |     0.196 |       -0.19  |     0.155 |     0.222 |   0.847 |
| ('rr_2025_12', False)    |       -0.163 |    -0.162 |    -0.162 |        0.01  |    -0.065 |    -0.081 |       -0.039 |    -0.108 |    -0.116 |        0.025 |    -0.098 |    -0.103 |   0.251 |
| ('rr_2026_02', True)     |        0.363 |     0.349 |     0.322 |        0.139 |     0.083 |     0.069 |        0.113 |     0.045 |     0.049 |        0.597 |     0.401 |     0.399 |   0.693 |
| ('rr_2026_03', True)     |        0.217 |     0.219 |     0.219 |        0.083 |     0.104 |     0.104 |        0.18  |     0.166 |     0.162 |        0.274 |     0.266 |     0.263 |   0.305 |
| ('rr_2026_04', True)     |        0.108 |     0.114 |     0.12  |        0.097 |     0.095 |     0.102 |        0.197 |     0.192 |     0.184 |        0.277 |     0.264 |     0.263 |   0.457 |
| ('rr_2026_05', True)     |       -0.206 |    -0.202 |    -0.201 |        0.095 |     0.089 |     0.094 |        0.098 |     0.083 |     0.08  |        0.068 |     0.009 |     0.012 |   0.449 |
| ('rr_2026_09_s1a', True) |       -0.568 |    -0.172 |    -0.101 |        0.253 |     0.295 |     0.327 |       -0.291 |    -0.111 |    -0.045 |       -0.207 |    -0.104 |    -0.046 |   0     |
| ('rr_2026_10_s1a', True) |       -0.642 |    -0.1   |    -0.024 |       -0.291 |    -0.181 |    -0.127 |       -0.302 |    -0.05  |     0.022 |       -0.149 |    -0.022 |     0.04  |   0.132 |

Per-target log error, period_end:

|                          |   A baseline |   A iv115 |   A point |   E baseline |   E iv115 |   E point |   G baseline |   G iv115 |   G point |   R baseline |   R iv115 |   R point |   naive |
|:-------------------------|-------------:|----------:|----------:|-------------:|----------:|----------:|-------------:|----------:|----------:|-------------:|----------:|----------:|--------:|
| ('rev_2025fy', True)     |       -0.034 |    -0.029 |    -0.017 |       -0.035 |    -0.033 |    -0.019 |        0.127 |     0.075 |     0.08  |        1.011 |     0.968 |     0.964 |   0.145 |
| ('rev_2025q3', True)     |       -0.315 |    -0.089 |    -0.01  |       -0.378 |    -0.091 |    -0.011 |       -0.252 |    -0.05  |     0.154 |        1.059 |     0.994 |     0.981 |   0.603 |
| ('rev_2026q1', True)     |       -0.021 |    -0.018 |    -0.015 |       -0.044 |    -0.039 |    -0.027 |        0.118 |     0.06  |     0.067 |        1.196 |     0.754 |     0.723 |   0.237 |
| ('rev_2026q2', True)     |       -0.12  |    -0.121 |    -0.122 |       -0.131 |    -0.129 |    -0.12  |        0.558 |     0.404 |     0.394 |        1.861 |     1.478 |     1.426 |   0.115 |
| ('rev_2026q3', True)     |       -0.324 |    -0.198 |    -0.129 |        0.138 |     0.129 |     0.165 |       -0.012 |     0.023 |     0.065 |       -0.013 |     0.01  |     0.039 |  -0.038 |
| ('rr_2025_10', True)     |       -0.44  |    -0.131 |    -0.012 |       -0.517 |    -0.133 |    -0.014 |       -0.113 |     0.132 |     0.196 |       -0.19  |     0.155 |     0.222 |   0.847 |
| ('rr_2025_12', True)     |        0.04  |     0.032 |     0.019 |        0.03  |    -0.012 |    -0.023 |        0.101 |    -0.004 |    -0.006 |        0.205 |     0.015 |     0.008 |   0.251 |
| ('rr_2026_02', True)     |        0.363 |     0.349 |     0.322 |        0.139 |     0.083 |     0.069 |        0.113 |     0.045 |     0.049 |        0.597 |     0.401 |     0.399 |   0.693 |
| ('rr_2026_03', True)     |        0.217 |     0.219 |     0.219 |        0.14  |     0.15  |     0.161 |        0.211 |     0.188 |     0.185 |        0.274 |     0.266 |     0.263 |   0.305 |
| ('rr_2026_04', True)     |        0.108 |     0.114 |     0.12  |        0.072 |     0.07  |     0.078 |        0.192 |     0.175 |     0.178 |        0.277 |     0.264 |     0.263 |   0.457 |
| ('rr_2026_05', True)     |       -0.206 |    -0.202 |    -0.201 |        0.151 |     0.14  |     0.143 |        0.098 |     0.083 |     0.08  |        0.004 |    -0.043 |    -0.048 |   0.449 |
| ('rr_2026_09_s1a', True) |       -0.494 |    -0.319 |    -0.217 |        0.234 |     0.213 |     0.267 |       -0.124 |    -0.064 |     0.008 |       -0.117 |    -0.073 |    -0.023 |   0     |
| ('rr_2026_10_s1a', True) |       -0.537 |    -0.324 |    -0.201 |       -0.413 |    -0.356 |    -0.291 |       -0.005 |     0.054 |     0.133 |        0.008 |     0.05  |     0.099 |   0.132 |

## OpenAI targets (ml_fixed second test set), mean |log error|

|                                             |   baseline |   point |   iv110 |   iv115 |   iv125 |
|:--------------------------------------------|-----------:|--------:|--------:|--------:|--------:|
| ('period_end', 'M1')                        |      0.063 |   0.141 |   0.118 |   0.101 |   0.087 |
| ('period_end', 'M2')                        |      0.063 |   0.146 |   0.124 |   0.113 |   0.099 |
| ('period_end', 'SS claims-only path')       |      0.112 |   0.205 |   0.188 |   0.179 |   0.161 |
| ('period_end', 'naive last claim (panel)')  |      0.254 |   0.255 |   0.255 |   0.254 |   0.254 |
| ('publication', 'M1')                       |      0.067 |   0.112 |   0.102 |   0.09  |   0.081 |
| ('publication', 'M2')                       |      0.079 |   0.102 |   0.092 |   0.087 |   0.083 |
| ('publication', 'SS claims-only path')      |      0.13  |   0.15  |   0.143 |   0.14  |   0.135 |
| ('publication', 'naive last claim (panel)') |      0.216 |   0.216 |   0.216 |   0.216 |   0.216 |

n targets:

|                                             |   baseline |   point |   iv110 |   iv115 |   iv125 |
|:--------------------------------------------|-----------:|--------:|--------:|--------:|--------:|
| ('period_end', 'M1')                        |         15 |      15 |      15 |      15 |      15 |
| ('period_end', 'M2')                        |         15 |      15 |      15 |      15 |      15 |
| ('period_end', 'SS claims-only path')       |         16 |      16 |      16 |      16 |      16 |
| ('period_end', 'naive last claim (panel)')  |         16 |      16 |      16 |      16 |      16 |
| ('publication', 'M1')                       |         15 |      15 |      15 |      15 |      15 |
| ('publication', 'M2')                       |         15 |      15 |      15 |      15 |      15 |
| ('publication', 'SS claims-only path')      |         16 |      16 |      16 |      16 |      16 |
| ('publication', 'naive last claim (panel)') |         16 |      16 |      16 |      16 |      16 |

## Baseline reruns vs the 2026-11-10 reports (max relative difference of the forecasts)

E differs because indicators.csv changed after its 2026-11-10 run (commit 7606f58 dropped the truncated
oldest OpenRouter week); every comparison here uses the fresh baseline, same code and data as the variants.

| rule        | model                       |   n_old |   n_new |   max_rel_diff |
|:------------|:----------------------------|--------:|--------:|---------------:|
| publication | A claims only               |      15 |      15 |       0.00e+00 |
| publication | B + indicators              |      15 |      15 |       0.00e+00 |
| publication | E OpenRouter tokens x price |      15 |      15 |       4.35e-02 |
| publication | G GitHub new-repo share     |      15 |      15 |       0.00e+00 |
| publication | R Ramp adoption             |      15 |      15 |       0.00e+00 |
| publication | Bayes + indicators          |      30 |      30 |       1.88e-04 |
| publication | ml_fixed (all rows)         |     390 |     390 |       0.00e+00 |
| period_end  | A claims only               |      13 |      13 |       0.00e+00 |
| period_end  | B + indicators              |      13 |      13 |       0.00e+00 |
| period_end  | E OpenRouter tokens x price |      13 |      13 |       1.32e-01 |
| period_end  | G GitHub new-repo share     |      13 |      13 |       0.00e+00 |
| period_end  | R Ramp adoption             |      13 |      13 |       0.00e+00 |
| period_end  | Bayes + indicators          |      26 |      26 |       2.29e-05 |
| period_end  | ml_fixed (all rows)         |     339 |     339 |       0.00e+00 |

## Q4 2026 registrations: Nov 30 / Dec 31 run-rate medians and Q4 revenue means ($B)

| set       | model   | asof       |   rr_2026-11-30 |   rr_2026-12-31 |   rr_2026-12-31 q10 |   rr_2026-12-31 q90 |   rev_q4_mean |   rev_q4 q10 |   rev_q4 q90 |
|:----------|:--------|:-----------|----------------:|----------------:|--------------------:|--------------------:|--------------:|-------------:|-------------:|
| q4_frozen | A       | 2026-11-12 |          184.92 |          220.21 |               99.68 |              486.49 |         49.93 |        24.02 |        82.37 |
| q4_frozen | B       | 2026-11-12 |          239.68 |          312.59 |              228.17 |              428.24 |         57.16 |        43.46 |        72.27 |
| q4_frozen | E       | 2026-11-12 |          169.07 |          217.77 |              127.29 |              372.57 |         38.48 |        26.59 |        52    |
| q4_frozen | G       | 2026-11-12 |          121.02 |          132.18 |               77.82 |              224.52 |         26.79 |        16.83 |        38.44 |
| q4_frozen | R       | 2026-11-12 |           99.69 |          105.18 |               69.31 |              159.6  |         22.64 |        16.06 |        30.05 |
| q4_frozen | P1      | 2026-11-12 |          146.68 |          170.64 |              170.64 |              170.64 |         34.14 |        34.14 |        34.14 |
| q4_frozen | P2      | 2026-11-12 |          136.39 |          155.83 |              155.83 |              155.83 |         29.19 |        29.19 |        29.19 |
| q4_frozen | N       | 2026-11-12 |           71    |           71    |               71    |               71    |         17.9  |        17.9  |        17.9  |
| q4_frozen | C       | 2026-11-12 |          138.76 |          159.7  |              159.7  |              159.7  |         32.09 |        32.09 |        32.09 |
| q4_v2     | A       | 2026-11-13 |           94.52 |           99.63 |               65.87 |              150.71 |         23.48 |        16.83 |        30.96 |
| q4_v2     | B       | 2026-11-13 |          152.17 |          189.02 |              146.52 |              243.83 |         37.79 |        30.39 |        45.8  |
| q4_v2     | E       | 2026-11-13 |          156.12 |          213.84 |              130.25 |              351.1  |         37.11 |        26.19 |        49.45 |
| q4_v2     | G       | 2026-11-13 |           90.54 |           95.82 |               64.47 |              142.43 |         20.84 |        14.49 |        28.05 |
| q4_v2     | R       | 2026-11-13 |           86.47 |           90.1  |               62.68 |              129.52 |         20.08 |        14.91 |        25.84 |
| q4_v2     | P1      | 2026-11-13 |          107.4  |          119.49 |              119.49 |              119.49 |         25.88 |        25.88 |        25.88 |
| q4_v2     | P2      | 2026-11-13 |          145.06 |          173.89 |              173.89 |              173.89 |         33.38 |        33.38 |        33.38 |
| q4_v2     | N       | 2026-11-13 |           71    |           71    |               71    |               71    |         17.9  |        17.9  |        17.9  |
| q4_v2     | C       | 2026-11-13 |          110.88 |          126.18 |              126.18 |              126.18 |         26.13 |        26.13 |        26.13 |
| q4_v2pub  | A       | 2026-11-13 |           87.58 |           95.69 |               75.1  |              121.93 |         21.94 |        18.96 |        25.09 |
| q4_v2pub  | B       | 2026-11-13 |           99.72 |          117.88 |               97.39 |              142.68 |         25.54 |        22.76 |        28.43 |
| q4_v2pub  | E       | 2026-11-13 |           91.42 |          102.09 |               74.96 |              139.03 |         22.93 |        19.14 |        26.99 |
| q4_v2pub  | G       | 2026-11-13 |           87.3  |           95.29 |               76.61 |              118.53 |         21.83 |        19.08 |        24.71 |
| q4_v2pub  | R       | 2026-11-13 |           86.7  |           94.36 |               76.23 |              116.79 |         21.66 |        18.98 |        24.48 |
| q4_v2pub  | P1      | 2026-11-13 |           96.04 |          115.62 |              115.62 |              115.62 |         23.73 |        23.73 |        23.73 |
| q4_v2pub  | P2      | 2026-11-13 |           95.19 |          114.11 |              114.11 |              114.11 |         23.6  |        23.6  |        23.6  |
| q4_v2pub  | N       | 2026-11-13 |           81    |           81    |               81    |               81    |         19.81 |        19.81 |        19.81 |
| q4_v2pub  | C       | 2026-11-13 |           89.58 |          100.05 |              100.05 |              100.05 |         22.38 |        22.38 |        22.38 |

Chart: `forecast_vs_actual_interval_2026-11-13.png`.
