# Filing numbers as evidence once public (sub-task O): backtest (2026-11-13)

Pre-registered in DATA_PLAN.md (子任务 O) before any run. Evidence rule **pub**: at each vintage every row
published on or before the vintage is evidence, including filing = 1 (S-1, S-1/A, press reports of the draft
S-1) and filing = 2 (third-party numbers published after a filing). The target's own origin stays out and
backtest.check_no_leak stays on. Everything else mirrors sub-task N (scripts/forecast_interval.py): same
frames, refit at every vintage, crossing position profiled, lower-claim rule iv115 (point as a sensitivity).
Comparator: N's no-filing files `forecast_interval_<model>_<variant>_<rule>_2026-11-13.csv`.
Errors are log(actual / forecast); + = forecast too low.

**The sample is tiny.** Only targets forecast after 2026-09-30 can change: 9 under the publication rule (the six
S-1 revenue numbers forecast on 2026-10-25, which can see only the Fortune draft-S-1 leak, plus rev_2026q3,
rr_2026_09_s1a, rr_2026_10_s1a) and 3 under the period_end rule. Every interval below over 3-9 targets, one
per target and highly correlated across models, is close to uninformative.

- publication: 15 common targets; changed (filing rows in the pub evidence): rev_2025fy, rev_2025q1, rev_2025q2, rev_2025q3, rev_2026q1, rev_2026q2, rev_2026q3, rr_2026_09_s1a, rr_2026_10_s1a; post-t0 (N's flags): rev_2026q3, rr_2025_10, rr_2026_02, rr_2026_03, rr_2026_04, rr_2026_05, rr_2026_09_s1a, rr_2026_10_s1a.
- period_end: 13 common targets; changed (filing rows in the pub evidence): rev_2026q3, rr_2026_09_s1a, rr_2026_10_s1a; post-t0 (N's flags): rev_2025fy, rev_2025q3, rev_2026q1, rev_2026q2, rev_2026q3, rr_2025_10, rr_2025_12, rr_2026_02, rr_2026_03, rr_2026_04, rr_2026_05, rr_2026_09_s1a, rr_2026_10_s1a.
- Naive: backtest._naive on each run's own evidence (taken from model A's iv115 file), so under pub it may
  carry forward a filing = 2 run-rate claim (semi74 / sacra70); for revenue it joins point run-rate claims and
  never reads a filing revenue number. N's eval used the naive of A [baseline]; here both naives come from
  A [iv115] (they differ from N's by the crossing position only: at most 0.021 in a revenue forecast).

## Reading (written after the runs)

- Publication rule, 9 changed targets: using the public filing numbers lowers mean |log error| for A (0.076 ->
  0.031), B (0.144 -> 0.060), E (0.098 -> 0.046) and Bayes (0.123 -> 0.059); the paired 90% intervals exclude
  zero for all four. G (0.034 -> 0.033) and R (0.055 -> 0.044) barely move: they were already close.
- H2 2026 (3 targets, publication): every model improves (A 0.157 -> 0.043, B 0.336 -> 0.114, Bayes 0.276 -> 0.111),
  but none beats the naive last claim, which also improves (0.056 -> 0.041). B and Bayes are still worse than the
  naive (paired +0.07, intervals above zero); A, E, G, R are level with it. B still puts the Oct 31 run-rate at
  90.7 (actual 81, outside its 80% interval) with Q3 GAAP, Sacra ~70, SemiAnalysis ~74 and the WSJ '>80' in hand.
- Period_end rule, 3 changed targets: mixed. All of the gain is rr_2026_10_s1a (A 112 -> 76.7, B 159 -> 115,
  E 116 -> 89.4, Bayes 145 -> 122). rev_2026q3 and rr_2026_09_s1a, whose only new input is Fortune's Q1-2026
  revenue (4.73), get slightly worse for most models (G rev_2026q3 +0.023 -> +0.085; Bayes rr_2026_09_s1a
  -0.439 -> -0.488). Only E's interval excludes zero on the improving side; G's excludes zero on the worse side.
- The six S-1 revenue targets forecast on 2026-10-25 see one new row, Fortune's Q1-2026 4.73 (2026-09-30): the
  error shrinks on 5 of 6 for A, B and E, 4 of 6 for R and Bayes, 3 of 6 for G. The moves are small (at most 0.016
  log for A, 0.053 for G).
- 80% coverage: publication A 0.73 -> 0.80, B 0.60 -> 0.67, E 0.87 -> 1.00, Bayes 0.80 -> 0.93, G and R
  unchanged; period_end unchanged for every model.
- Point sensitivity (A B E G R): same direction. On the publication rule's changed targets B (-0.066), E (-0.050)
  and R (-0.011) gain about as much as under iv115; A gains half as much (-0.021, interval touches zero) because
  A's no-filing point run was already closer; G is flat (+0.005).
- Every target without a filing row in its pub evidence reproduces N's forecast mean bit for bit (check below).
- Caveats: (a) rr_2026_10_semi (filing = 2) may be Q3 GAAP annualized (its notes say so); with rev_2026q3 also in
  the evidence for the run-rate targets that double-counts one number, and it is what the pub naive carries
  forward for rr_2026_10_s1a (74). (b) rev_2026q1's value 4.729 is noted as derived with S-1/A No. 1 numbers
  (2026-11-02) but dated 2026-10-26, so it enters rev_2026q3's evidence (vintage 2026-11-01). Replacing it with the
  10-26-derivable 16.3 - 11.555 = 4.745 moves model A's rev_2026q3 forecast from 17.831 to 17.833 and the run-rate
  forecasts by < 0.01%: immaterial, but the row's date is generous. (c) 9 / 3 targets, one per number, strongly
  correlated across models: the intervals are descriptive, not evidence of a general effect.

## Mean |log error| by model x evidence x rule, iv115 (common targets)

| rule        | model                       | evidence   |   n_all |   mae_all |   bias_all |   n_post |   mae_post |   n_h2 |   mae_h2 |   n_chg |   mae_chg |   bias_chg |   cov80 |   cov80_chg |   cov50 |
|:------------|:----------------------------|:-----------|--------:|----------:|-----------:|---------:|-----------:|-------:|---------:|--------:|----------:|-----------:|--------:|------------:|--------:|
| period_end  | A claims only               | nofiling   |      13 |     0.165 |     -0.055 |       13 |      0.165 |      3 |    0.28  |       3 |     0.28  |     -0.28  |   0.846 |       1     |   0.385 |
| period_end  | A claims only               | pub        |      13 |     0.145 |     -0.027 |       13 |      0.145 |      3 |    0.196 |       3 |     0.196 |     -0.159 |   0.846 |       1     |   0.462 |
| period_end  | B + indicators              | nofiling   |      13 |     0.202 |     -0.161 |       13 |      0.202 |      3 |    0.484 |       3 |     0.484 |     -0.484 |   0.615 |       0     |   0.462 |
| period_end  | B + indicators              | pub        |      13 |     0.179 |     -0.137 |       13 |      0.179 |      3 |    0.382 |       3 |     0.382 |     -0.382 |   0.615 |       0     |   0.462 |
| period_end  | E OpenRouter tokens x price | nofiling   |      13 |     0.121 |     -0     |       13 |      0.121 |      3 |    0.233 |       3 |     0.233 |     -0.005 |   0.923 |       1     |   0.538 |
| period_end  | E OpenRouter tokens x price | pub        |      13 |     0.1   |      0.018 |       13 |      0.1   |      3 |    0.141 |       3 |     0.141 |      0.075 |   0.923 |       1     |   0.615 |
| period_end  | G GitHub new-repo share     | nofiling   |      13 |     0.104 |      0.086 |       13 |      0.104 |      3 |    0.047 |       3 |     0.047 |      0.004 |   0.923 |       1     |   0.769 |
| period_end  | G GitHub new-repo share     | pub        |      13 |     0.113 |      0.094 |       13 |      0.113 |      3 |    0.082 |       3 |     0.082 |      0.039 |   0.923 |       1     |   0.769 |
| period_end  | R Ramp adoption             | nofiling   |      13 |     0.421 |      0.403 |       13 |      0.421 |      3 |    0.044 |       3 |     0.044 |     -0.005 |   0.615 |       1     |   0.462 |
| period_end  | R Ramp adoption             | pub        |      13 |     0.422 |      0.405 |       13 |      0.422 |      3 |    0.05  |       3 |     0.05  |      0.004 |   0.615 |       1     |   0.462 |
| period_end  | Bayes + indicators          | nofiling   |      13 |     0.184 |     -0.167 |       13 |      0.184 |      3 |    0.439 |       3 |     0.439 |     -0.439 |   0.769 |       0     |   0.538 |
| period_end  | Bayes + indicators          | pub        |      13 |     0.175 |     -0.158 |       13 |      0.175 |      3 |    0.402 |       3 |     0.402 |     -0.402 |   0.769 |       0     |   0.538 |
| period_end  | N naive last claim          | nofiling   |      13 |     0.329 |      0.323 |       13 |      0.329 |      3 |    0.056 |       3 |     0.056 |      0.031 | nan     |     nan     | nan     |
| period_end  | N naive last claim          | pub        |      13 |     0.33  |      0.324 |       13 |      0.33  |      3 |    0.061 |       3 |     0.061 |      0.036 | nan     |     nan     | nan     |
| publication | A claims only               | nofiling   |      15 |     0.124 |     -0.005 |        8 |      0.186 |      3 |    0.157 |       9 |     0.076 |     -0.028 |   0.733 |       0.889 |   0.467 |
| publication | A claims only               | pub        |      15 |     0.097 |      0.021 |        8 |      0.143 |      3 |    0.043 |       9 |     0.031 |      0.014 |   0.8   |       1     |   0.6   |
| publication | B + indicators              | nofiling   |      15 |     0.139 |     -0.067 |        8 |      0.21  |      3 |    0.336 |       9 |     0.144 |     -0.081 |   0.6   |       0.667 |   0.333 |
| publication | B + indicators              | pub        |      15 |     0.089 |     -0.032 |        8 |      0.127 |      3 |    0.114 |       9 |     0.06  |     -0.023 |   0.667 |       0.778 |   0.467 |
| publication | E OpenRouter tokens x price | nofiling   |      15 |     0.097 |      0.046 |        8 |      0.13  |      3 |    0.179 |       9 |     0.098 |      0.057 |   0.867 |       0.778 |   0.467 |
| publication | E OpenRouter tokens x price | pub        |      15 |     0.065 |      0.034 |        8 |      0.08  |      3 |    0.045 |       9 |     0.046 |      0.038 |   1     |       1     |   0.533 |
| publication | G GitHub new-repo share     | nofiling   |      15 |     0.069 |      0.019 |        8 |      0.1   |      3 |    0.061 |       9 |     0.034 |     -0.025 |   0.933 |       1     |   0.733 |
| publication | G GitHub new-repo share     | pub        |      15 |     0.068 |      0.042 |        8 |      0.088 |      3 |    0.028 |       9 |     0.033 |      0.014 |   0.933 |       1     |   0.8   |
| publication | R Ramp adoption             | nofiling   |      15 |     0.113 |      0.073 |        8 |      0.155 |      3 |    0.049 |       9 |     0.055 |      0.011 |   0.933 |       1     |   0.733 |
| publication | R Ramp adoption             | pub        |      15 |     0.106 |      0.07  |        8 |      0.15  |      3 |    0.034 |       9 |     0.044 |      0.005 |   0.933 |       1     |   0.667 |
| publication | Bayes + indicators          | nofiling   |      15 |     0.117 |     -0.065 |        8 |      0.171 |      3 |    0.276 |       9 |     0.123 |     -0.061 |   0.8   |       0.667 |   0.467 |
| publication | Bayes + indicators          | pub        |      15 |     0.079 |     -0.041 |        8 |      0.11  |      3 |    0.111 |       9 |     0.059 |     -0.022 |   0.933 |       0.889 |   0.6   |
| publication | N naive last claim          | nofiling   |      15 |     0.263 |      0.258 |        8 |      0.365 |      3 |    0.056 |       9 |     0.104 |      0.096 | nan     |     nan     | nan     |
| publication | N naive last claim          | pub        |      15 |     0.26  |      0.255 |        8 |      0.359 |      3 |    0.041 |       9 |     0.099 |      0.092 | nan     |     nan     | nan     |

## Paired differences in |log error|, iv115 (`B vs A`: negative = B better; 90% bootstrap interval)

On `all` and `post-t0` the unchanged targets contribute exact zeros to `pub vs nofiling`; `changed` is the
informative subset.

### publication

| subset   | comparison                                                                  |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:---------|:----------------------------------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| all      | A claims only [pub] vs A claims only [nofiling]                             |  15 |      -0.027 |    -0.049 |    -0.007 |          8 |       1 |
| all      | A claims only [pub] vs N naive last claim [pub]                             |  15 |      -0.162 |    -0.245 |    -0.091 |         13 |       2 |
| all      | A claims only [nofiling] vs N naive last claim [nofiling]                   |  15 |      -0.139 |    -0.23  |    -0.055 |         13 |       2 |
| all      | B + indicators [pub] vs B + indicators [nofiling]                           |  15 |      -0.05  |    -0.092 |    -0.015 |          8 |       1 |
| all      | B + indicators [pub] vs N naive last claim [pub]                            |  15 |      -0.171 |    -0.275 |    -0.078 |         12 |       3 |
| all      | B + indicators [nofiling] vs N naive last claim [nofiling]                  |  15 |      -0.124 |    -0.247 |    -0.006 |         12 |       3 |
| all      | E OpenRouter tokens x price [pub] vs E OpenRouter tokens x price [nofiling] |  15 |      -0.031 |    -0.069 |    -0.002 |          7 |       2 |
| all      | E OpenRouter tokens x price [pub] vs N naive last claim [pub]               |  15 |      -0.194 |    -0.292 |    -0.107 |         13 |       2 |
| all      | E OpenRouter tokens x price [nofiling] vs N naive last claim [nofiling]     |  15 |      -0.166 |    -0.274 |    -0.062 |         12 |       3 |
| all      | G GitHub new-repo share [pub] vs G GitHub new-repo share [nofiling]         |  15 |      -0     |    -0.015 |     0.012 |          5 |       4 |
| all      | G GitHub new-repo share [pub] vs N naive last claim [pub]                   |  15 |      -0.191 |    -0.288 |    -0.106 |         14 |       1 |
| all      | G GitHub new-repo share [nofiling] vs N naive last claim [nofiling]         |  15 |      -0.194 |    -0.291 |    -0.106 |         14 |       1 |
| all      | R Ramp adoption [pub] vs R Ramp adoption [nofiling]                         |  15 |      -0.007 |    -0.02  |     0.005 |          6 |       3 |
| all      | R Ramp adoption [pub] vs N naive last claim [pub]                           |  15 |      -0.154 |    -0.239 |    -0.081 |         14 |       1 |
| all      | R Ramp adoption [nofiling] vs N naive last claim [nofiling]                 |  15 |      -0.15  |    -0.237 |    -0.075 |         14 |       1 |
| all      | Bayes + indicators [pub] vs Bayes + indicators [nofiling]                   |  15 |      -0.038 |    -0.07  |    -0.01  |          7 |       2 |
| all      | Bayes + indicators [pub] vs N naive last claim [pub]                        |  15 |      -0.181 |    -0.289 |    -0.084 |         13 |       2 |
| all      | Bayes + indicators [nofiling] vs N naive last claim [nofiling]              |  15 |      -0.146 |    -0.267 |    -0.032 |         12 |       3 |
| all      | N naive last claim [pub] vs N naive last claim [nofiling]                   |  15 |      -0.003 |    -0.008 |     0     |          2 |       0 |
| post-t0  | A claims only [pub] vs A claims only [nofiling]                             |   8 |      -0.043 |    -0.081 |    -0.005 |          3 |       0 |
| post-t0  | A claims only [pub] vs N naive last claim [pub]                             |   8 |      -0.216 |    -0.361 |    -0.087 |          6 |       2 |
| post-t0  | A claims only [nofiling] vs N naive last claim [nofiling]                   |   8 |      -0.179 |    -0.344 |    -0.024 |          6 |       2 |
| post-t0  | B + indicators [pub] vs B + indicators [nofiling]                           |   8 |      -0.083 |    -0.152 |    -0.017 |          3 |       0 |
| post-t0  | B + indicators [pub] vs N naive last claim [pub]                            |   8 |      -0.233 |    -0.417 |    -0.06  |          5 |       3 |
| post-t0  | B + indicators [nofiling] vs N naive last claim [nofiling]                  |   8 |      -0.155 |    -0.385 |     0.067 |          5 |       3 |
| post-t0  | E OpenRouter tokens x price [pub] vs E OpenRouter tokens x price [nofiling] |   8 |      -0.05  |    -0.116 |     0.005 |          2 |       1 |
| post-t0  | E OpenRouter tokens x price [pub] vs N naive last claim [pub]               |   8 |      -0.279 |    -0.438 |    -0.128 |          6 |       2 |
| post-t0  | E OpenRouter tokens x price [nofiling] vs N naive last claim [nofiling]     |   8 |      -0.235 |    -0.423 |    -0.046 |          5 |       3 |
| post-t0  | G GitHub new-repo share [pub] vs G GitHub new-repo share [nofiling]         |   8 |      -0.012 |    -0.034 |     0.001 |          2 |       1 |
| post-t0  | G GitHub new-repo share [pub] vs N naive last claim [pub]                   |   8 |      -0.272 |    -0.432 |    -0.124 |          7 |       1 |
| post-t0  | G GitHub new-repo share [nofiling] vs N naive last claim [nofiling]         |   8 |      -0.265 |    -0.429 |    -0.111 |          7 |       1 |
| post-t0  | R Ramp adoption [pub] vs R Ramp adoption [nofiling]                         |   8 |      -0.006 |    -0.025 |     0.009 |          2 |       1 |
| post-t0  | R Ramp adoption [pub] vs N naive last claim [pub]                           |   8 |      -0.21  |    -0.356 |    -0.081 |          7 |       1 |
| post-t0  | R Ramp adoption [nofiling] vs N naive last claim [nofiling]                 |   8 |      -0.21  |    -0.359 |    -0.076 |          7 |       1 |
| post-t0  | Bayes + indicators [pub] vs Bayes + indicators [nofiling]                   |   8 |      -0.062 |    -0.116 |    -0.008 |          3 |       0 |
| post-t0  | Bayes + indicators [pub] vs N naive last claim [pub]                        |   8 |      -0.25  |    -0.438 |    -0.071 |          6 |       2 |
| post-t0  | Bayes + indicators [nofiling] vs N naive last claim [nofiling]              |   8 |      -0.194 |    -0.415 |     0.018 |          5 |       3 |
| post-t0  | N naive last claim [pub] vs N naive last claim [nofiling]                   |   8 |      -0.006 |    -0.016 |     0     |          2 |       0 |
| H2-2026  | A claims only [pub] vs A claims only [nofiling]                             |   3 |      -0.114 |    -0.152 |    -0.075 |          3 |       0 |
| H2-2026  | A claims only [pub] vs N naive last claim [pub]                             |   3 |       0.002 |    -0.014 |     0.017 |          1 |       2 |
| H2-2026  | A claims only [nofiling] vs N naive last claim [nofiling]                   |   3 |       0.1   |     0.033 |     0.168 |          1 |       2 |
| H2-2026  | B + indicators [pub] vs B + indicators [nofiling]                           |   3 |      -0.222 |    -0.271 |    -0.174 |          3 |       0 |
| H2-2026  | B + indicators [pub] vs N naive last claim [pub]                            |   3 |       0.072 |     0.035 |     0.11  |          0 |       3 |
| H2-2026  | B + indicators [nofiling] vs N naive last claim [nofiling]                  |   3 |       0.279 |     0.263 |     0.296 |          0 |       3 |
| H2-2026  | E OpenRouter tokens x price [pub] vs E OpenRouter tokens x price [nofiling] |   3 |      -0.134 |    -0.243 |    -0.025 |          2 |       1 |
| H2-2026  | E OpenRouter tokens x price [pub] vs N naive last claim [pub]               |   3 |       0.004 |    -0.044 |     0.052 |          1 |       2 |
| H2-2026  | E OpenRouter tokens x price [nofiling] vs N naive last claim [nofiling]     |   3 |       0.123 |     0.033 |     0.213 |          0 |       3 |
| H2-2026  | G GitHub new-repo share [pub] vs G GitHub new-repo share [nofiling]         |   3 |      -0.033 |    -0.065 |    -0.001 |          2 |       1 |
| H2-2026  | G GitHub new-repo share [pub] vs N naive last claim [pub]                   |   3 |      -0.013 |    -0.032 |     0.006 |          2 |       1 |
| H2-2026  | G GitHub new-repo share [nofiling] vs N naive last claim [nofiling]         |   3 |       0.004 |    -0.06  |     0.069 |          2 |       1 |
| H2-2026  | R Ramp adoption [pub] vs R Ramp adoption [nofiling]                         |   3 |      -0.015 |    -0.054 |     0.024 |          2 |       1 |
| H2-2026  | R Ramp adoption [pub] vs N naive last claim [pub]                           |   3 |      -0.007 |    -0.025 |     0.011 |          2 |       1 |
| H2-2026  | R Ramp adoption [nofiling] vs N naive last claim [nofiling]                 |   3 |      -0.007 |    -0.078 |     0.064 |          2 |       1 |
| H2-2026  | Bayes + indicators [pub] vs Bayes + indicators [nofiling]                   |   3 |      -0.165 |    -0.221 |    -0.109 |          3 |       0 |
| H2-2026  | Bayes + indicators [pub] vs N naive last claim [pub]                        |   3 |       0.069 |     0.011 |     0.128 |          1 |       2 |
| H2-2026  | Bayes + indicators [nofiling] vs N naive last claim [nofiling]              |   3 |       0.219 |     0.192 |     0.246 |          0 |       3 |
| H2-2026  | N naive last claim [pub] vs N naive last claim [nofiling]                   |   3 |      -0.015 |    -0.029 |    -0.002 |          2 |       0 |
| changed  | A claims only [pub] vs A claims only [nofiling]                             |   9 |      -0.045 |    -0.078 |    -0.015 |          8 |       1 |
| changed  | A claims only [pub] vs N naive last claim [pub]                             |   9 |      -0.068 |    -0.099 |    -0.036 |          7 |       2 |
| changed  | A claims only [nofiling] vs N naive last claim [nofiling]                   |   9 |      -0.028 |    -0.084 |     0.033 |          7 |       2 |
| changed  | B + indicators [pub] vs B + indicators [nofiling]                           |   9 |      -0.084 |    -0.144 |    -0.03  |          8 |       1 |
| changed  | B + indicators [pub] vs N naive last claim [pub]                            |   9 |      -0.039 |    -0.086 |     0.01  |          6 |       3 |
| changed  | B + indicators [nofiling] vs N naive last claim [nofiling]                  |   9 |       0.039 |    -0.048 |     0.13  |          6 |       3 |
| changed  | E OpenRouter tokens x price [pub] vs E OpenRouter tokens x price [nofiling] |   9 |      -0.052 |    -0.109 |    -0.003 |          7 |       2 |
| changed  | E OpenRouter tokens x price [pub] vs N naive last claim [pub]               |   9 |      -0.053 |    -0.085 |    -0.019 |          7 |       2 |
| changed  | E OpenRouter tokens x price [nofiling] vs N naive last claim [nofiling]     |   9 |      -0.006 |    -0.063 |     0.065 |          6 |       3 |
| changed  | G GitHub new-repo share [pub] vs G GitHub new-repo share [nofiling]         |   9 |      -0.001 |    -0.024 |     0.02  |          5 |       4 |
| changed  | G GitHub new-repo share [pub] vs N naive last claim [pub]                   |   9 |      -0.066 |    -0.092 |    -0.038 |          8 |       1 |
| changed  | G GitHub new-repo share [nofiling] vs N naive last claim [nofiling]         |   9 |      -0.07  |    -0.108 |    -0.025 |          8 |       1 |
| changed  | R Ramp adoption [pub] vs R Ramp adoption [nofiling]                         |   9 |      -0.012 |    -0.031 |     0.008 |          6 |       3 |
| changed  | R Ramp adoption [pub] vs N naive last claim [pub]                           |   9 |      -0.056 |    -0.081 |    -0.03  |          8 |       1 |
| changed  | R Ramp adoption [nofiling] vs N naive last claim [nofiling]                 |   9 |      -0.049 |    -0.078 |    -0.013 |          8 |       1 |
| changed  | Bayes + indicators [pub] vs Bayes + indicators [nofiling]                   |   9 |      -0.064 |    -0.111 |    -0.019 |          7 |       2 |
| changed  | Bayes + indicators [pub] vs N naive last claim [pub]                        |   9 |      -0.04  |    -0.086 |     0.012 |          7 |       2 |
| changed  | Bayes + indicators [nofiling] vs N naive last claim [nofiling]              |   9 |       0.019 |    -0.056 |     0.098 |          6 |       3 |
| changed  | N naive last claim [pub] vs N naive last claim [nofiling]                   |   9 |      -0.005 |    -0.014 |     0     |          2 |       0 |

### period_end

| subset   | comparison                                                                  |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:---------|:----------------------------------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| all      | A claims only [pub] vs A claims only [nofiling]                             |  13 |      -0.02  |    -0.061 |     0.002 |          1 |       2 |
| all      | A claims only [pub] vs N naive last claim [pub]                             |  13 |      -0.184 |    -0.303 |    -0.067 |         10 |       3 |
| all      | A claims only [nofiling] vs N naive last claim [nofiling]                   |  13 |      -0.164 |    -0.292 |    -0.038 |          9 |       4 |
| all      | B + indicators [pub] vs B + indicators [nofiling]                           |  13 |      -0.023 |    -0.073 |     0.003 |          2 |       1 |
| all      | B + indicators [pub] vs N naive last claim [pub]                            |  13 |      -0.151 |    -0.31  |     0.007 |          9 |       4 |
| all      | B + indicators [nofiling] vs N naive last claim [nofiling]                  |  13 |      -0.127 |    -0.301 |     0.047 |          9 |       4 |
| all      | E OpenRouter tokens x price [pub] vs E OpenRouter tokens x price [nofiling] |  13 |      -0.021 |    -0.059 |    -0     |          3 |       0 |
| all      | E OpenRouter tokens x price [pub] vs N naive last claim [pub]               |  13 |      -0.23  |    -0.353 |    -0.111 |         10 |       3 |
| all      | E OpenRouter tokens x price [nofiling] vs N naive last claim [nofiling]     |  13 |      -0.207 |    -0.34  |    -0.078 |          9 |       4 |
| all      | G GitHub new-repo share [pub] vs G GitHub new-repo share [nofiling]         |  13 |       0.008 |    -0     |     0.018 |          1 |       2 |
| all      | G GitHub new-repo share [pub] vs N naive last claim [pub]                   |  13 |      -0.217 |    -0.349 |    -0.089 |         10 |       3 |
| all      | G GitHub new-repo share [nofiling] vs N naive last claim [nofiling]         |  13 |      -0.224 |    -0.353 |    -0.098 |         11 |       2 |
| all      | R Ramp adoption [pub] vs R Ramp adoption [nofiling]                         |  13 |       0.001 |    -0.001 |     0.005 |          2 |       1 |
| all      | R Ramp adoption [pub] vs N naive last claim [pub]                           |  13 |       0.092 |    -0.138 |     0.347 |          8 |       5 |
| all      | R Ramp adoption [nofiling] vs N naive last claim [nofiling]                 |  13 |       0.092 |    -0.138 |     0.347 |          8 |       5 |
| all      | Bayes + indicators [pub] vs Bayes + indicators [nofiling]                   |  13 |      -0.009 |    -0.036 |     0.009 |          1 |       2 |
| all      | Bayes + indicators [pub] vs N naive last claim [pub]                        |  13 |      -0.155 |    -0.321 |     0.01  |          9 |       4 |
| all      | Bayes + indicators [nofiling] vs N naive last claim [nofiling]              |  13 |      -0.145 |    -0.317 |     0.024 |          9 |       4 |
| all      | N naive last claim [pub] vs N naive last claim [nofiling]                   |  13 |       0.001 |     0     |     0.003 |          0 |       1 |
| post-t0  | A claims only [pub] vs A claims only [nofiling]                             |  13 |      -0.02  |    -0.061 |     0.002 |          1 |       2 |
| post-t0  | A claims only [pub] vs N naive last claim [pub]                             |  13 |      -0.184 |    -0.303 |    -0.067 |         10 |       3 |
| post-t0  | A claims only [nofiling] vs N naive last claim [nofiling]                   |  13 |      -0.164 |    -0.292 |    -0.038 |          9 |       4 |
| post-t0  | B + indicators [pub] vs B + indicators [nofiling]                           |  13 |      -0.023 |    -0.073 |     0.003 |          2 |       1 |
| post-t0  | B + indicators [pub] vs N naive last claim [pub]                            |  13 |      -0.151 |    -0.31  |     0.007 |          9 |       4 |
| post-t0  | B + indicators [nofiling] vs N naive last claim [nofiling]                  |  13 |      -0.127 |    -0.301 |     0.047 |          9 |       4 |
| post-t0  | E OpenRouter tokens x price [pub] vs E OpenRouter tokens x price [nofiling] |  13 |      -0.021 |    -0.059 |    -0     |          3 |       0 |
| post-t0  | E OpenRouter tokens x price [pub] vs N naive last claim [pub]               |  13 |      -0.23  |    -0.353 |    -0.111 |         10 |       3 |
| post-t0  | E OpenRouter tokens x price [nofiling] vs N naive last claim [nofiling]     |  13 |      -0.207 |    -0.34  |    -0.078 |          9 |       4 |
| post-t0  | G GitHub new-repo share [pub] vs G GitHub new-repo share [nofiling]         |  13 |       0.008 |    -0     |     0.018 |          1 |       2 |
| post-t0  | G GitHub new-repo share [pub] vs N naive last claim [pub]                   |  13 |      -0.217 |    -0.349 |    -0.089 |         10 |       3 |
| post-t0  | G GitHub new-repo share [nofiling] vs N naive last claim [nofiling]         |  13 |      -0.224 |    -0.353 |    -0.098 |         11 |       2 |
| post-t0  | R Ramp adoption [pub] vs R Ramp adoption [nofiling]                         |  13 |       0.001 |    -0.001 |     0.005 |          2 |       1 |
| post-t0  | R Ramp adoption [pub] vs N naive last claim [pub]                           |  13 |       0.092 |    -0.138 |     0.347 |          8 |       5 |
| post-t0  | R Ramp adoption [nofiling] vs N naive last claim [nofiling]                 |  13 |       0.092 |    -0.138 |     0.347 |          8 |       5 |
| post-t0  | Bayes + indicators [pub] vs Bayes + indicators [nofiling]                   |  13 |      -0.009 |    -0.036 |     0.009 |          1 |       2 |
| post-t0  | Bayes + indicators [pub] vs N naive last claim [pub]                        |  13 |      -0.155 |    -0.321 |     0.01  |          9 |       4 |
| post-t0  | Bayes + indicators [nofiling] vs N naive last claim [nofiling]              |  13 |      -0.145 |    -0.317 |     0.024 |          9 |       4 |
| post-t0  | N naive last claim [pub] vs N naive last claim [nofiling]                   |  13 |       0.001 |     0     |     0.003 |          0 |       1 |
| H2-2026  | A claims only [pub] vs A claims only [nofiling]                             |   3 |      -0.085 |    -0.179 |     0.009 |          1 |       2 |
| H2-2026  | A claims only [pub] vs N naive last claim [pub]                             |   3 |       0.134 |    -0.003 |     0.271 |          1 |       2 |
| H2-2026  | A claims only [nofiling] vs N naive last claim [nofiling]                   |   3 |       0.224 |     0.171 |     0.277 |          0 |       3 |
| H2-2026  | B + indicators [pub] vs B + indicators [nofiling]                           |   3 |      -0.102 |    -0.215 |     0.012 |          2 |       1 |
| H2-2026  | B + indicators [pub] vs N naive last claim [pub]                            |   3 |       0.321 |     0.227 |     0.415 |          0 |       3 |
| H2-2026  | B + indicators [nofiling] vs N naive last claim [nofiling]                  |   3 |       0.428 |     0.328 |     0.527 |          0 |       3 |
| H2-2026  | E OpenRouter tokens x price [pub] vs E OpenRouter tokens x price [nofiling] |   3 |      -0.092 |    -0.177 |    -0.006 |          3 |       0 |
| H2-2026  | E OpenRouter tokens x price [pub] vs N naive last claim [pub]               |   3 |       0.08  |    -0.007 |     0.167 |          1 |       2 |
| H2-2026  | E OpenRouter tokens x price [nofiling] vs N naive last claim [nofiling]     |   3 |       0.176 |     0.132 |     0.22  |          0 |       3 |
| H2-2026  | G GitHub new-repo share [pub] vs G GitHub new-repo share [nofiling]         |   3 |       0.035 |     0.014 |     0.056 |          1 |       2 |
| H2-2026  | G GitHub new-repo share [pub] vs N naive last claim [pub]                   |   3 |       0.021 |    -0.017 |     0.059 |          1 |       2 |
| H2-2026  | G GitHub new-repo share [nofiling] vs N naive last claim [nofiling]         |   3 |      -0.009 |    -0.057 |     0.038 |          2 |       1 |
| H2-2026  | R Ramp adoption [pub] vs R Ramp adoption [nofiling]                         |   3 |       0.005 |    -0.005 |     0.015 |          2 |       1 |
| H2-2026  | R Ramp adoption [pub] vs N naive last claim [pub]                           |   3 |      -0.012 |    -0.058 |     0.035 |          2 |       1 |
| H2-2026  | R Ramp adoption [nofiling] vs N naive last claim [nofiling]                 |   3 |      -0.012 |    -0.064 |     0.04  |          2 |       1 |
| H2-2026  | Bayes + indicators [pub] vs Bayes + indicators [nofiling]                   |   3 |      -0.037 |    -0.113 |     0.038 |          1 |       2 |
| H2-2026  | Bayes + indicators [pub] vs N naive last claim [pub]                        |   3 |       0.341 |     0.265 |     0.416 |          0 |       3 |
| H2-2026  | Bayes + indicators [nofiling] vs N naive last claim [nofiling]              |   3 |       0.383 |     0.318 |     0.448 |          0 |       3 |
| H2-2026  | N naive last claim [pub] vs N naive last claim [nofiling]                   |   3 |       0.005 |     0     |     0.009 |          0 |       1 |
| changed  | A claims only [pub] vs A claims only [nofiling]                             |   3 |      -0.085 |    -0.179 |     0.009 |          1 |       2 |
| changed  | A claims only [pub] vs N naive last claim [pub]                             |   3 |       0.134 |    -0.003 |     0.271 |          1 |       2 |
| changed  | A claims only [nofiling] vs N naive last claim [nofiling]                   |   3 |       0.224 |     0.171 |     0.277 |          0 |       3 |
| changed  | B + indicators [pub] vs B + indicators [nofiling]                           |   3 |      -0.102 |    -0.215 |     0.012 |          2 |       1 |
| changed  | B + indicators [pub] vs N naive last claim [pub]                            |   3 |       0.321 |     0.227 |     0.415 |          0 |       3 |
| changed  | B + indicators [nofiling] vs N naive last claim [nofiling]                  |   3 |       0.428 |     0.328 |     0.527 |          0 |       3 |
| changed  | E OpenRouter tokens x price [pub] vs E OpenRouter tokens x price [nofiling] |   3 |      -0.092 |    -0.177 |    -0.006 |          3 |       0 |
| changed  | E OpenRouter tokens x price [pub] vs N naive last claim [pub]               |   3 |       0.08  |    -0.007 |     0.167 |          1 |       2 |
| changed  | E OpenRouter tokens x price [nofiling] vs N naive last claim [nofiling]     |   3 |       0.176 |     0.132 |     0.22  |          0 |       3 |
| changed  | G GitHub new-repo share [pub] vs G GitHub new-repo share [nofiling]         |   3 |       0.035 |     0.014 |     0.056 |          1 |       2 |
| changed  | G GitHub new-repo share [pub] vs N naive last claim [pub]                   |   3 |       0.021 |    -0.017 |     0.059 |          1 |       2 |
| changed  | G GitHub new-repo share [nofiling] vs N naive last claim [nofiling]         |   3 |      -0.009 |    -0.057 |     0.038 |          2 |       1 |
| changed  | R Ramp adoption [pub] vs R Ramp adoption [nofiling]                         |   3 |       0.005 |    -0.005 |     0.015 |          2 |       1 |
| changed  | R Ramp adoption [pub] vs N naive last claim [pub]                           |   3 |      -0.012 |    -0.058 |     0.035 |          2 |       1 |
| changed  | R Ramp adoption [nofiling] vs N naive last claim [nofiling]                 |   3 |      -0.012 |    -0.064 |     0.04  |          2 |       1 |
| changed  | Bayes + indicators [pub] vs Bayes + indicators [nofiling]                   |   3 |      -0.037 |    -0.113 |     0.038 |          1 |       2 |
| changed  | Bayes + indicators [pub] vs N naive last claim [pub]                        |   3 |       0.341 |     0.265 |     0.416 |          0 |       3 |
| changed  | Bayes + indicators [nofiling] vs N naive last claim [nofiling]              |   3 |       0.383 |     0.318 |     0.448 |          0 |       3 |
| changed  | N naive last claim [pub] vs N naive last claim [nofiling]                   |   3 |       0.005 |     0     |     0.009 |          0 |       1 |

## Changed targets, publication, iv115

| target         | vintage    |   actual | filing rows in the pub evidence, by publication date                                                                                                                                                                                                                         |   naive no-filing | naive pub (input)                                                                                                                      |
|:---------------|:-----------|---------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------:|:---------------------------------------------------------------------------------------------------------------------------------------|
| rev_2025fy     | 2026-10-25 |    4.59  | 2026-09-30: rev_2026q1_fortune                                                                                                                                                                                                                                               |             4.08  | 4.08 (interp rr_2025_01=1@2025-01-16;rr_2025_05=3@2025-05-28;rr_2025_10=7@2025-10-30;rr_2025_12=9@2025-12-28;rr_2026_02=14@2026-02-09) |
| rev_2025q1     | 2026-10-25 |    0.376 | 2026-09-30: rev_2026q1_fortune                                                                                                                                                                                                                                               |             0.325 | 0.325 (interp rr_2025_01=1@2025-01-16;rr_2025_05=3@2025-05-28)                                                                         |
| rev_2025q2     | 2026-10-25 |    0.787 | 2026-09-30: rev_2026q1_fortune                                                                                                                                                                                                                                               |             0.678 | 0.678 (interp rr_2025_01=1@2025-01-16;rr_2025_05=3@2025-05-28;rr_2025_10=7@2025-10-30)                                                 |
| rev_2025q3     | 2026-10-25 |    1.382 | 2026-09-30: rev_2026q1_fortune                                                                                                                                                                                                                                               |             1.18  | 1.18 (interp rr_2025_05=3@2025-05-28;rr_2025_10=7@2025-10-30)                                                                          |
| rev_2026q1     | 2026-10-25 |    4.729 | 2026-09-30: rev_2026q1_fortune                                                                                                                                                                                                                                               |             4.1   | 4.1 (interp rr_2025_12=9@2025-12-28;rr_2026_02=14@2026-02-09;rr_2026_03=19@2026-03-02;rr_2026_04=30@2026-04-03)                        |
| rev_2026q2     | 2026-10-25 |   11.555 | 2026-09-30: rev_2026q1_fortune                                                                                                                                                                                                                                               |            10.9   | 10.9 (interp rr_2026_03=19@2026-03-02;rr_2026_04=30@2026-04-03;rr_2026_05=47@2026-05-26;rr_2026_07_funda=71@2026-07-21)                |
| rev_2026q3     | 2026-11-01 |   16.96  | 2026-09-30: rev_2026q1_fortune; 2026-10-26: rev_2025fy, rev_2025q1, rev_2025q2, rev_2025q3, rev_2026q1, rev_2026q2; 2026-10-31: rr_2026_10_sacra70 (B E G R bayes also: 2026-10-26: seg_cloud_h1_2026, seg_sub_h1_2026)                                                      |            17.6   | 17.5 (interp rr_2026_05=47@2026-05-26;rr_2026_07_funda=71@2026-07-21;rr_2026_10_sacra70=70@2026-10-16)                                 |
| rr_2026_09_s1a | 2026-11-09 |   71     | 2026-09-30: rev_2026q1_fortune; 2026-10-26: rev_2025fy, rev_2025q1, rev_2025q2, rev_2025q3, rev_2026q1, rev_2026q2; 2026-10-31: rr_2026_10_sacra70; 2026-11-02: rev_2026q3; 2026-11-03: rr_2026_10_semi (B E G R bayes also: 2026-10-26: seg_cloud_h1_2026, seg_sub_h1_2026) |            71     | 71 (rr_2026_07_funda=71@2026-07-21)                                                                                                    |
| rr_2026_10_s1a | 2026-11-09 |   81     | 2026-09-30: rev_2026q1_fortune; 2026-10-26: rev_2025fy, rev_2025q1, rev_2025q2, rev_2025q3, rev_2026q1, rev_2026q2; 2026-10-31: rr_2026_10_sacra70; 2026-11-02: rev_2026q3; 2026-11-03: rr_2026_10_semi (B E G R bayes also: 2026-10-26: seg_cloud_h1_2026, seg_sub_h1_2026) |            71     | 74 (rr_2026_10_semi=74@2026-10-17)                                                                                                     |

Forecast no-filing -> pub (log error no-filing -> pub; in 80% interval no-filing/pub):

| target         |   actual | A                                         | B                                          | E                                          | G                                          | R                                          | bayes                                      |
|:---------------|---------:|:------------------------------------------|:-------------------------------------------|:-------------------------------------------|:-------------------------------------------|:-------------------------------------------|:-------------------------------------------|
| rev_2025fy     |    4.59  | 4.39 -> 4.45 (+0.045 -> +0.030; in80 YY)  | 4.29 -> 4.4 (+0.067 -> +0.042; in80 YY)    | 4.24 -> 4.31 (+0.078 -> +0.062; in80 YY)   | 4.65 -> 4.54 (-0.013 -> +0.010; in80 YY)   | 4.28 -> 4.4 (+0.071 -> +0.042; in80 YY)    | 4.29 -> 4.4 (+0.068 -> +0.043; in80 YY)    |
| rev_2025q1     |    0.376 | 0.354 -> 0.36 (+0.059 -> +0.045; in80 YY) | 0.37 -> 0.379 (+0.016 -> -0.008; in80 YY)  | 0.339 -> 0.344 (+0.104 -> +0.088; in80 YY) | 0.388 -> 0.385 (-0.031 -> -0.025; in80 YY) | 0.404 -> 0.423 (-0.071 -> -0.119; in80 YY) | 0.372 -> 0.381 (+0.011 -> -0.014; in80 YY) |
| rev_2025q2     |    0.787 | 0.749 -> 0.76 (+0.050 -> +0.035; in80 YY) | 0.734 -> 0.752 (+0.070 -> +0.046; in80 YY) | 0.729 -> 0.741 (+0.077 -> +0.061; in80 YY) | 0.782 -> 0.755 (+0.006 -> +0.042; in80 YY) | 0.732 -> 0.754 (+0.073 -> +0.043; in80 YY) | 0.735 -> 0.753 (+0.068 -> +0.044; in80 YY) |
| rev_2025q3     |    1.382 | 1.34 -> 1.36 (+0.032 -> +0.017; in80 YY)  | 1.28 -> 1.31 (+0.078 -> +0.054; in80 YY)   | 1.31 -> 1.34 (+0.050 -> +0.034; in80 YY)   | 1.37 -> 1.3 (+0.008 -> +0.060; in80 YY)    | 1.28 -> 1.31 (+0.076 -> +0.051; in80 YY)   | 1.27 -> 1.3 (+0.082 -> +0.060; in80 YY)    |
| rev_2026q1     |    4.729 | 4.59 -> 4.66 (+0.030 -> +0.014; in80 YY)  | 4.49 -> 4.61 (+0.052 -> +0.025; in80 YY)   | 4.58 -> 4.66 (+0.033 -> +0.015; in80 YY)   | 5.02 -> 4.9 (-0.059 -> -0.035; in80 YY)    | 4.49 -> 4.61 (+0.052 -> +0.025; in80 YY)   | 4.5 -> 4.62 (+0.050 -> +0.023; in80 YY)    |
| rev_2026q2     |   11.555 | 11.6 -> 11.7 (-0.001 -> -0.013; in80 YY)  | 11.6 -> 11.8 (-0.003 -> -0.023; in80 YY)   | 11.6 -> 11.8 (-0.005 -> -0.017; in80 YY)   | 11.5 -> 11.1 (+0.005 -> +0.044; in80 YY)   | 11.5 -> 11.7 (+0.007 -> -0.009; in80 YY)   | 11.6 -> 11.8 (-0.001 -> -0.020; in80 YY)   |
| rev_2026q3     |   16.96  | 20.7 -> 17.8 (-0.198 -> -0.050; in80 YY)  | 23 -> 20.1 (-0.303 -> -0.168; in80 NN)     | 15.9 -> 15.3 (+0.062 -> +0.106; in80 YY)   | 16.6 -> 17.1 (+0.022 -> -0.008; in80 YY)   | 16.6 -> 17.3 (+0.021 -> -0.019; in80 YY)   | 22 -> 20.7 (-0.259 -> -0.197; in80 NN)     |
| rr_2026_09_s1a |   71     | 84.3 -> 72.2 (-0.172 -> -0.017; in80 NY)  | 96.9 -> 75.4 (-0.311 -> -0.060; in80 NY)   | 52.8 -> 70.2 (+0.295 -> +0.011; in80 NY)   | 79.3 -> 72.5 (-0.111 -> -0.021; in80 YY)   | 78.8 -> 72.7 (-0.104 -> -0.024; in80 YY)   | 91.9 -> 75 (-0.258 -> -0.055; in80 NY)     |
| rr_2026_10_s1a |   81     | 89.6 -> 76.2 (-0.100 -> +0.061; in80 YY)  | 120 -> 90.7 (-0.393 -> -0.113; in80 NN)    | 97 -> 82.6 (-0.181 -> -0.019; in80 NY)     | 85.1 -> 76.7 (-0.050 -> +0.054; in80 YY)   | 82.8 -> 76.3 (-0.022 -> +0.060; in80 YY)   | 110 -> 87.7 (-0.309 -> -0.080; in80 NY)    |

## Changed targets, period_end, iv115

| target         | vintage    |   actual | filing rows in the pub evidence, by publication date                                                                                                                                                                    |   naive no-filing | naive pub (input)                                                     |
|:---------------|:-----------|---------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------:|:----------------------------------------------------------------------|
| rev_2026q3     | 2026-09-30 |    16.96 | 2026-09-30: rev_2026q1_fortune                                                                                                                                                                                          |              17.6 | 17.6 (interp rr_2026_05=47@2026-05-26;rr_2026_07_funda=71@2026-07-21) |
| rr_2026_09_s1a | 2026-09-30 |    71    | 2026-09-30: rev_2026q1_fortune                                                                                                                                                                                          |              71   | 71 (rr_2026_07_funda=71@2026-07-21)                                   |
| rr_2026_10_s1a | 2026-10-31 |    81    | 2026-09-30: rev_2026q1_fortune; 2026-10-26: rev_2025fy, rev_2025q1, rev_2025q2, rev_2025q3, rev_2026q1, rev_2026q2; 2026-10-31: rr_2026_10_sacra70 (B E G R bayes also: 2026-10-26: seg_cloud_h1_2026, seg_sub_h1_2026) |              71   | 70 (rr_2026_10_sacra70=70@2026-10-16)                                 |

Forecast no-filing -> pub (log error no-filing -> pub; in 80% interval no-filing/pub):

| target         |   actual | A                                        | B                                      | E                                        | G                                        | R                                        | bayes                                    |
|:---------------|---------:|:-----------------------------------------|:---------------------------------------|:-----------------------------------------|:-----------------------------------------|:-----------------------------------------|:-----------------------------------------|
| rev_2026q3     |    16.96 | 20.7 -> 21 (-0.198 -> -0.212; in80 YY)   | 22.5 -> 23 (-0.283 -> -0.303; in80 NN) | 14.9 -> 15.2 (+0.129 -> +0.112; in80 YY) | 16.6 -> 15.6 (+0.023 -> +0.085; in80 YY) | 16.8 -> 16.9 (+0.010 -> +0.006; in80 YY) | 22.8 -> 23.1 (-0.296 -> -0.311; in80 NN) |
| rr_2026_09_s1a |    71    | 97.7 -> 97.8 (-0.319 -> -0.320; in80 YY) | 116 -> 116 (-0.495 -> -0.490; in80 NN) | 57.4 -> 57.4 (+0.213 -> +0.213; in80 YY) | 75.7 -> 75.7 (-0.064 -> -0.064; in80 YY) | 76.4 -> 76 (-0.073 -> -0.068; in80 YY)   | 110 -> 116 (-0.439 -> -0.488; in80 NN)   |
| rr_2026_10_s1a |    81    | 112 -> 76.7 (-0.324 -> +0.055; in80 YY)  | 159 -> 115 (-0.675 -> -0.354; in80 NN) | 116 -> 89.4 (-0.356 -> -0.099; in80 YY)  | 76.7 -> 73.5 (+0.054 -> +0.097; in80 YY) | 77.1 -> 75.2 (+0.050 -> +0.075; in80 YY) | 145 -> 122 (-0.584 -> -0.407; in80 NN)   |

## Sensitivity: lower claims as points (point), A B E G R (N ran no bayes point)

| rule        | model                       | evidence   |   n_all |   mae_all |   n_post |   mae_post |   n_h2 |   mae_h2 |   n_chg |   mae_chg |   cov80 |   cov80_chg |
|:------------|:----------------------------|:-----------|--------:|----------:|---------:|-----------:|-------:|---------:|--------:|----------:|--------:|------------:|
| period_end  | A claims only               | nofiling   |      13 |     0.123 |       13 |      0.123 |      3 |    0.182 |       3 |     0.182 |   0.846 |       1     |
| period_end  | A claims only               | pub        |      13 |     0.115 |       13 |      0.115 |      3 |    0.145 |       3 |     0.145 |   0.846 |       1     |
| period_end  | B + indicators              | nofiling   |      13 |     0.178 |       13 |      0.178 |      3 |    0.431 |       3 |     0.431 |   0.615 |       0     |
| period_end  | B + indicators              | pub        |      13 |     0.157 |       13 |      0.157 |      3 |    0.342 |       3 |     0.342 |   0.615 |       0     |
| period_end  | E OpenRouter tokens x price | nofiling   |      13 |     0.107 |       13 |      0.107 |      3 |    0.241 |       3 |     0.241 |   0.769 |       0.333 |
| period_end  | E OpenRouter tokens x price | pub        |      13 |     0.09  |       13 |      0.09  |      3 |    0.168 |       3 |     0.168 |   0.769 |       0.333 |
| period_end  | G GitHub new-repo share     | nofiling   |      13 |     0.123 |       13 |      0.123 |      3 |    0.069 |       3 |     0.069 |   0.923 |       1     |
| period_end  | G GitHub new-repo share     | pub        |      13 |     0.124 |       13 |      0.124 |      3 |    0.076 |       3 |     0.076 |   0.923 |       1     |
| period_end  | R Ramp adoption             | nofiling   |      13 |     0.42  |       13 |      0.42  |      3 |    0.054 |       3 |     0.054 |   0.615 |       1     |
| period_end  | R Ramp adoption             | pub        |      13 |     0.419 |       13 |      0.419 |      3 |    0.049 |       3 |     0.049 |   0.615 |       1     |
| period_end  | N naive last claim          | nofiling   |      13 |     0.329 |       13 |      0.329 |      3 |    0.056 |       3 |     0.056 | nan     |     nan     |
| period_end  | N naive last claim          | pub        |      13 |     0.33  |       13 |      0.33  |      3 |    0.061 |       3 |     0.061 | nan     |     nan     |
| publication | A claims only               | nofiling   |      15 |     0.101 |        8 |      0.141 |      3 |    0.085 |       9 |     0.053 |   0.8   |       1     |
| publication | A claims only               | pub        |      15 |     0.088 |        8 |      0.123 |      3 |    0.037 |       9 |     0.032 |   0.8   |       1     |
| publication | B + indicators              | nofiling   |      15 |     0.123 |        8 |      0.18  |      3 |    0.282 |       9 |     0.124 |   0.6   |       0.667 |
| publication | B + indicators              | pub        |      15 |     0.083 |        8 |      0.113 |      3 |    0.102 |       9 |     0.058 |   0.667 |       0.778 |
| publication | E OpenRouter tokens x price | nofiling   |      15 |     0.091 |        8 |      0.117 |      3 |    0.185 |       9 |     0.1   |   0.933 |       0.889 |
| publication | E OpenRouter tokens x price | pub        |      15 |     0.061 |        8 |      0.066 |      3 |    0.048 |       9 |     0.05  |   0.933 |       0.889 |
| publication | G GitHub new-repo share     | nofiling   |      15 |     0.072 |        8 |      0.1   |      3 |    0.043 |       9 |     0.033 |   0.933 |       1     |
| publication | G GitHub new-repo share     | pub        |      15 |     0.076 |        8 |      0.095 |      3 |    0.03  |       9 |     0.038 |   0.933 |       1     |
| publication | R Ramp adoption             | nofiling   |      15 |     0.118 |        8 |      0.162 |      3 |    0.045 |       9 |     0.056 |   0.933 |       1     |
| publication | R Ramp adoption             | pub        |      15 |     0.111 |        8 |      0.156 |      3 |    0.03  |       9 |     0.045 |   0.933 |       1     |
| publication | N naive last claim          | nofiling   |      15 |     0.263 |        8 |      0.365 |      3 |    0.056 |       9 |     0.104 | nan     |     nan     |
| publication | N naive last claim          | pub        |      15 |     0.26  |        8 |      0.359 |      3 |    0.041 |       9 |     0.099 | nan     |     nan     |

| subset   | rule        | comparison                                                                  |   n |   mean_diff |   ci90_lo |   ci90_hi |   improved |   worse |
|:---------|:------------|:----------------------------------------------------------------------------|----:|------------:|----------:|----------:|-----------:|--------:|
| changed  | publication | A claims only [pub] vs A claims only [nofiling]                             |   9 |      -0.021 |    -0.046 |     0.002 |          7 |       2 |
| changed  | publication | A claims only [pub] vs N naive last claim [pub]                             |   9 |      -0.067 |    -0.094 |    -0.039 |          8 |       1 |
| changed  | publication | A claims only [nofiling] vs N naive last claim [nofiling]                   |   9 |      -0.052 |    -0.094 |    -0.006 |          7 |       2 |
| changed  | publication | B + indicators [pub] vs B + indicators [nofiling]                           |   9 |      -0.066 |    -0.115 |    -0.02  |          7 |       2 |
| changed  | publication | B + indicators [pub] vs N naive last claim [pub]                            |   9 |      -0.041 |    -0.083 |     0.004 |          6 |       3 |
| changed  | publication | B + indicators [nofiling] vs N naive last claim [nofiling]                  |   9 |       0.02  |    -0.057 |     0.099 |          6 |       3 |
| changed  | publication | E OpenRouter tokens x price [pub] vs E OpenRouter tokens x price [nofiling] |   9 |      -0.05  |    -0.112 |    -0.001 |          7 |       2 |
| changed  | publication | E OpenRouter tokens x price [pub] vs N naive last claim [pub]               |   9 |      -0.049 |    -0.081 |    -0.013 |          7 |       2 |
| changed  | publication | E OpenRouter tokens x price [nofiling] vs N naive last claim [nofiling]     |   9 |      -0.004 |    -0.064 |     0.073 |          7 |       2 |
| changed  | publication | G GitHub new-repo share [pub] vs G GitHub new-repo share [nofiling]         |   9 |       0.005 |    -0.015 |     0.025 |          4 |       5 |
| changed  | publication | G GitHub new-repo share [pub] vs N naive last claim [pub]                   |   9 |      -0.061 |    -0.087 |    -0.035 |          8 |       1 |
| changed  | publication | G GitHub new-repo share [nofiling] vs N naive last claim [nofiling]         |   9 |      -0.071 |    -0.103 |    -0.036 |          7 |       2 |
| changed  | publication | R Ramp adoption [pub] vs R Ramp adoption [nofiling]                         |   9 |      -0.011 |    -0.026 |     0.005 |          7 |       2 |
| changed  | publication | R Ramp adoption [pub] vs N naive last claim [pub]                           |   9 |      -0.054 |    -0.077 |    -0.031 |          8 |       1 |
| changed  | publication | R Ramp adoption [nofiling] vs N naive last claim [nofiling]                 |   9 |      -0.048 |    -0.071 |    -0.023 |          7 |       2 |
| changed  | publication | N naive last claim [pub] vs N naive last claim [nofiling]                   |   9 |      -0.005 |    -0.014 |     0     |          2 |       0 |
| changed  | period_end  | A claims only [pub] vs A claims only [nofiling]                             |   3 |      -0.037 |    -0.083 |     0.009 |          1 |       2 |
| changed  | period_end  | A claims only [pub] vs N naive last claim [pub]                             |   3 |       0.084 |    -0.013 |     0.181 |          1 |       2 |
| changed  | period_end  | A claims only [nofiling] vs N naive last claim [nofiling]                   |   3 |       0.126 |     0.077 |     0.175 |          0 |       3 |
| changed  | period_end  | B + indicators [pub] vs B + indicators [nofiling]                           |   3 |      -0.089 |    -0.189 |     0.01  |          2 |       1 |
| changed  | period_end  | B + indicators [pub] vs N naive last claim [pub]                            |   3 |       0.281 |     0.198 |     0.363 |          0 |       3 |
| changed  | period_end  | B + indicators [nofiling] vs N naive last claim [nofiling]                  |   3 |       0.375 |     0.284 |     0.465 |          0 |       3 |
| changed  | period_end  | E OpenRouter tokens x price [pub] vs E OpenRouter tokens x price [nofiling] |   3 |      -0.073 |    -0.143 |    -0.004 |          3 |       0 |
| changed  | period_end  | E OpenRouter tokens x price [pub] vs N naive last claim [pub]               |   3 |       0.106 |    -0.004 |     0.217 |          1 |       2 |
| changed  | period_end  | E OpenRouter tokens x price [nofiling] vs N naive last claim [nofiling]     |   3 |       0.185 |     0.138 |     0.231 |          0 |       3 |
| changed  | period_end  | G GitHub new-repo share [pub] vs G GitHub new-repo share [nofiling]         |   3 |       0.008 |    -0.016 |     0.031 |          2 |       1 |
| changed  | period_end  | G GitHub new-repo share [pub] vs N naive last claim [pub]                   |   3 |       0.015 |    -0.022 |     0.052 |          1 |       2 |
| changed  | period_end  | G GitHub new-repo share [nofiling] vs N naive last claim [nofiling]         |   3 |       0.012 |     0.003 |     0.021 |          0 |       3 |
| changed  | period_end  | R Ramp adoption [pub] vs R Ramp adoption [nofiling]                         |   3 |      -0.005 |    -0.006 |    -0.004 |          3 |       0 |
| changed  | period_end  | R Ramp adoption [pub] vs N naive last claim [pub]                           |   3 |      -0.013 |    -0.037 |     0.011 |          2 |       1 |
| changed  | period_end  | R Ramp adoption [nofiling] vs N naive last claim [nofiling]                 |   3 |      -0.003 |    -0.022 |     0.016 |          1 |       2 |
| changed  | period_end  | N naive last claim [pub] vs N naive last claim [nofiling]                   |   3 |       0.005 |     0     |     0.009 |          0 |       1 |

Changed targets, publication, point: forecast no-filing -> pub

| target         |   actual | A                                          | B                                          | E                                          | G                                          | R                                          |
|:---------------|---------:|:-------------------------------------------|:-------------------------------------------|:-------------------------------------------|:-------------------------------------------|:-------------------------------------------|
| rev_2025fy     |    4.59  | 4.38 -> 4.42 (+0.047 -> +0.037; in80 YY)   | 4.31 -> 4.4 (+0.062 -> +0.043; in80 YY)    | 4.25 -> 4.28 (+0.078 -> +0.069; in80 YY)   | 4.65 -> 4.52 (-0.012 -> +0.015; in80 YY)   | 4.27 -> 4.39 (+0.072 -> +0.046; in80 YY)   |
| rev_2025q1     |    0.376 | 0.361 -> 0.364 (+0.042 -> +0.032; in80 YY) | 0.375 -> 0.382 (+0.002 -> -0.017; in80 YY) | 0.346 -> 0.349 (+0.084 -> +0.075; in80 YY) | 0.391 -> 0.386 (-0.040 -> -0.027; in80 YY) | 0.406 -> 0.425 (-0.078 -> -0.123; in80 YY) |
| rev_2025q2     |    0.787 | 0.745 -> 0.752 (+0.055 -> +0.045; in80 YY) | 0.732 -> 0.747 (+0.072 -> +0.052; in80 YY) | 0.726 -> 0.733 (+0.080 -> +0.072; in80 YY) | 0.778 -> 0.749 (+0.012 -> +0.049; in80 YY) | 0.727 -> 0.749 (+0.079 -> +0.050; in80 YY) |
| rev_2025q3     |    1.382 | 1.3 -> 1.32 (+0.058 -> +0.048; in80 YY)    | 1.26 -> 1.29 (+0.090 -> +0.070; in80 YY)   | 1.27 -> 1.29 (+0.081 -> +0.072; in80 YY)   | 1.34 -> 1.28 (+0.028 -> +0.077; in80 YY)   | 1.26 -> 1.29 (+0.092 -> +0.069; in80 YY)   |
| rev_2026q1     |    4.729 | 4.65 -> 4.69 (+0.016 -> +0.009; in80 YY)   | 4.54 -> 4.63 (+0.041 -> +0.020; in80 YY)   | 4.64 -> 4.69 (+0.018 -> +0.009; in80 YY)   | 5.08 -> 4.91 (-0.072 -> -0.038; in80 YY)   | 4.53 -> 4.63 (+0.043 -> +0.022; in80 YY)   |
| rev_2026q2     |   11.555 | 11.5 -> 11.6 (+0.000 -> -0.007; in80 YY)   | 11.6 -> 11.8 (-0.001 -> -0.017; in80 YY)   | 11.6 -> 11.7 (-0.003 -> -0.010; in80 YY)   | 11.5 -> 11 (+0.006 -> +0.048; in80 YY)     | 11.5 -> 11.6 (+0.008 -> -0.006; in80 YY)   |
| rev_2026q3     |   16.96  | 19.3 -> 17.4 (-0.129 -> -0.026; in80 YY)   | 22.2 -> 19.7 (-0.268 -> -0.151; in80 NN)   | 15.3 -> 14.9 (+0.101 -> +0.129; in80 YN)   | 15.9 -> 16.8 (+0.062 -> +0.012; in80 YY)   | 16.2 -> 16.9 (+0.049 -> +0.002; in80 YY)   |
| rr_2026_09_s1a |   71     | 78.6 -> 71.8 (-0.101 -> -0.012; in80 YY)   | 91.4 -> 74.9 (-0.253 -> -0.053; in80 NY)   | 51.2 -> 70.4 (+0.327 -> +0.008; in80 NY)   | 74.2 -> 72 (-0.045 -> -0.014; in80 YY)     | 74.3 -> 72.3 (-0.046 -> -0.018; in80 YY)   |
| rr_2026_10_s1a |   81     | 82.9 -> 75.3 (-0.024 -> +0.073; in80 YY)   | 112 -> 89.7 (-0.326 -> -0.102; in80 NN)    | 92 -> 80.5 (-0.127 -> +0.006; in80 YY)     | 79.3 -> 75.9 (+0.022 -> +0.065; in80 YY)   | 77.8 -> 75.5 (+0.040 -> +0.070; in80 YY)   |

Changed targets, period_end, point: forecast no-filing -> pub

| target         |   actual | A                                        | B                                        | E                                        | G                                        | R                                        |
|:---------------|---------:|:-----------------------------------------|:-----------------------------------------|:-----------------------------------------|:-----------------------------------------|:-----------------------------------------|
| rev_2026q3     |    16.96 | 19.3 -> 19.5 (-0.129 -> -0.140; in80 YY) | 21.7 -> 22.1 (-0.246 -> -0.263; in80 NN) | 14.4 -> 14.5 (+0.165 -> +0.153; in80 NN) | 15.9 -> 15.1 (+0.065 -> +0.115; in80 YY) | 16.3 -> 16.4 (+0.039 -> +0.033; in80 YY) |
| rr_2026_09_s1a |    71    | 88.2 -> 88.5 (-0.217 -> -0.220; in80 YY) | 110 -> 109 (-0.436 -> -0.432; in80 NN)   | 54.3 -> 54.4 (+0.267 -> +0.267; in80 NN) | 70.4 -> 70.9 (+0.008 -> +0.001; in80 YY) | 72.7 -> 72.4 (-0.023 -> -0.019; in80 YY) |
| rr_2026_10_s1a |    81    | 99 -> 75.2 (-0.201 -> +0.074; in80 YY)   | 149 -> 113 (-0.611 -> -0.330; in80 NN)   | 108 -> 87.9 (-0.291 -> -0.082; in80 YY)  | 70.9 -> 72.4 (+0.133 -> +0.113; in80 YY) | 73.4 -> 73.8 (+0.099 -> +0.094; in80 YY) |

## Check: unchanged targets identical to N's no-filing run

Targets with no filing row in their pub evidence must reproduce N exactly. `max rel diff sd` on revenue
targets comes from the orchestrator's uncommitted model.py fix (observation_predictive scales revenue noise
by the forecast mean instead of the target value), which postdates N's runs; it never touches the mean.
`revenue sd vs fix-predicted` applies that change to N's sd (sqrt(sd^2 + 0.003^2 (forecast^2 - actual^2)))
and compares it with the pub run: ~1e-9 means the whole sd difference is the fix.

| rule        | model                       | variant   | targets nofiling/pub   | only in one   |   n unchanged-evidence |   max rel diff forecast |   max rel diff sd | sd diff on                                     |   revenue sd vs fix-predicted (max rel) |   run-rate sd (max rel diff) | in80 flips   |   n with filing rows | with filing rows but same forecast   |
|:------------|:----------------------------|:----------|:-----------------------|:--------------|-----------------------:|------------------------:|------------------:|:-----------------------------------------------|----------------------------------------:|-----------------------------:|:-------------|---------------------:|:-------------------------------------|
| period_end  | A claims only               | iv115     | 13/13                  | -             |                     10 |                0.00e+00 |          1.61e-05 | rev_2025q3, rev_2025fy, rev_2026q1, rev_2026q2 |                                4.44e-16 |                     0.00e+00 | -            |                    3 | -                                    |
| publication | A claims only               | iv115     | 15/15                  | -             |                      6 |                0.00e+00 |          0.00e+00 | -                                              |                              nan        |                     0.00e+00 | -            |                    9 | -                                    |
| period_end  | A claims only               | point     | 13/13                  | -             |                     10 |                0.00e+00 |          1.63e-05 | rev_2025q3, rev_2025fy, rev_2026q1, rev_2026q2 |                                4.44e-16 |                     0.00e+00 | -            |                    3 | -                                    |
| publication | A claims only               | point     | 15/15                  | -             |                      6 |                0.00e+00 |          0.00e+00 | -                                              |                              nan        |                     0.00e+00 | -            |                    9 | -                                    |
| period_end  | B + indicators              | iv115     | 13/13                  | -             |                     10 |                0.00e+00 |          2.20e-05 | rev_2025q3, rev_2025fy, rev_2026q1, rev_2026q2 |                                3.33e-16 |                     0.00e+00 | -            |                    3 | -                                    |
| publication | B + indicators              | iv115     | 15/15                  | -             |                      6 |                0.00e+00 |          0.00e+00 | -                                              |                              nan        |                     0.00e+00 | -            |                    9 | -                                    |
| period_end  | B + indicators              | point     | 13/13                  | -             |                     10 |                0.00e+00 |          2.21e-05 | rev_2025q3, rev_2025fy, rev_2026q1, rev_2026q2 |                                4.44e-16 |                     0.00e+00 | -            |                    3 | -                                    |
| publication | B + indicators              | point     | 15/15                  | -             |                      6 |                0.00e+00 |          0.00e+00 | -                                              |                              nan        |                     0.00e+00 | -            |                    9 | -                                    |
| period_end  | Bayes + indicators          | iv115     | 13/13                  | -             |                     10 |                0.00e+00 |          1.79e-05 | rev_2025q3, rev_2025fy, rev_2026q1, rev_2026q2 |                                5.60e-07 |                     0.00e+00 | -            |                    3 | -                                    |
| publication | Bayes + indicators          | iv115     | 15/15                  | -             |                      6 |                0.00e+00 |          0.00e+00 | -                                              |                              nan        |                     0.00e+00 | -            |                    9 | -                                    |
| period_end  | E OpenRouter tokens x price | iv115     | 13/13                  | -             |                     10 |                0.00e+00 |          1.87e-05 | rev_2025q3, rev_2025fy, rev_2026q1, rev_2026q2 |                                1.22e-15 |                     0.00e+00 | -            |                    3 | -                                    |
| publication | E OpenRouter tokens x price | iv115     | 15/15                  | -             |                      6 |                0.00e+00 |          0.00e+00 | -                                              |                              nan        |                     0.00e+00 | -            |                    9 | -                                    |
| period_end  | E OpenRouter tokens x price | point     | 13/13                  | -             |                     10 |                0.00e+00 |          1.76e-05 | rev_2025q3, rev_2025fy, rev_2026q1, rev_2026q2 |                                3.33e-16 |                     0.00e+00 | -            |                    3 | -                                    |
| publication | E OpenRouter tokens x price | point     | 15/15                  | -             |                      6 |                0.00e+00 |          0.00e+00 | -                                              |                              nan        |                     0.00e+00 | -            |                    9 | -                                    |
| period_end  | G GitHub new-repo share     | iv115     | 13/13                  | -             |                     10 |                0.00e+00 |          3.53e-05 | rev_2025q3, rev_2025fy, rev_2026q1, rev_2026q2 |                                4.44e-16 |                     0.00e+00 | -            |                    3 | -                                    |
| publication | G GitHub new-repo share     | iv115     | 15/15                  | -             |                      6 |                0.00e+00 |          0.00e+00 | -                                              |                              nan        |                     0.00e+00 | -            |                    9 | -                                    |
| period_end  | G GitHub new-repo share     | point     | 13/13                  | -             |                     10 |                0.00e+00 |          3.45e-05 | rev_2025q3, rev_2025fy, rev_2026q1, rev_2026q2 |                                3.33e-16 |                     0.00e+00 | -            |                    3 | -                                    |
| publication | G GitHub new-repo share     | point     | 15/15                  | -             |                      6 |                0.00e+00 |          0.00e+00 | -                                              |                              nan        |                     0.00e+00 | -            |                    9 | -                                    |
| period_end  | R Ramp adoption             | iv115     | 13/13                  | -             |                     10 |                0.00e+00 |          6.35e-04 | rev_2025q3, rev_2025fy, rev_2026q1, rev_2026q2 |                                3.33e-16 |                     0.00e+00 | -            |                    3 | -                                    |
| publication | R Ramp adoption             | iv115     | 15/15                  | -             |                      6 |                0.00e+00 |          0.00e+00 | -                                              |                              nan        |                     0.00e+00 | -            |                    9 | -                                    |
| period_end  | R Ramp adoption             | point     | 13/13                  | -             |                     10 |                0.00e+00 |          5.48e-04 | rev_2025q3, rev_2025fy, rev_2026q1, rev_2026q2 |                                4.44e-16 |                     0.00e+00 | -            |                    3 | -                                    |
| publication | R Ramp adoption             | point     | 15/15                  | -             |                      6 |                0.00e+00 |          0.00e+00 | -                                              |                              nan        |                     0.00e+00 | -            |                    9 | -                                    |

## Leak checks

1. Filing rows in the evidence, all 22 pub files: 608 (file, target, row) entries, 0 violations of published <= vintage < target publication (dates re-read from observations.csv, not the deduplicated table). Per target and vintage (union over models):

| target         | vintage    | target published   | filing rows (filing flag) by publication date                                                                                                                                                                                                                                               | latest published   | all <= vintage   |   pub files |
|:---------------|:-----------|:-------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------|:-----------------|------------:|
| rev_2026q3     | 2026-09-30 | 2026-11-02         | 2026-09-30: rev_2026q1_fortune (1)                                                                                                                                                                                                                                                          | 2026-09-30         | True             |          11 |
| rr_2026_09_s1a | 2026-09-30 | 2026-11-10         | 2026-09-30: rev_2026q1_fortune (1)                                                                                                                                                                                                                                                          | 2026-09-30         | True             |          11 |
| rev_2025fy     | 2026-10-25 | 2026-10-26         | 2026-09-30: rev_2026q1_fortune (1)                                                                                                                                                                                                                                                          | 2026-09-30         | True             |          11 |
| rev_2025q1     | 2026-10-25 | 2026-10-26         | 2026-09-30: rev_2026q1_fortune (1)                                                                                                                                                                                                                                                          | 2026-09-30         | True             |          11 |
| rev_2025q2     | 2026-10-25 | 2026-10-26         | 2026-09-30: rev_2026q1_fortune (1)                                                                                                                                                                                                                                                          | 2026-09-30         | True             |          11 |
| rev_2025q3     | 2026-10-25 | 2026-10-26         | 2026-09-30: rev_2026q1_fortune (1)                                                                                                                                                                                                                                                          | 2026-09-30         | True             |          11 |
| rev_2026q1     | 2026-10-25 | 2026-10-26         | 2026-09-30: rev_2026q1_fortune (1)                                                                                                                                                                                                                                                          | 2026-09-30         | True             |          11 |
| rev_2026q2     | 2026-10-25 | 2026-10-26         | 2026-09-30: rev_2026q1_fortune (1)                                                                                                                                                                                                                                                          | 2026-09-30         | True             |          11 |
| rr_2026_10_s1a | 2026-10-31 | 2026-11-10         | 2026-09-30: rev_2026q1_fortune (1); 2026-10-26: rev_2025fy (1), rev_2025q1 (1), rev_2025q2 (1), rev_2025q3 (1), rev_2026q1 (1), rev_2026q2 (1), seg_cloud_h1_2026 (1), seg_sub_h1_2026 (1); 2026-10-31: rr_2026_10_sacra70 (2)                                                              | 2026-10-31         | True             |          11 |
| rev_2026q3     | 2026-11-01 | 2026-11-02         | 2026-09-30: rev_2026q1_fortune (1); 2026-10-26: rev_2025fy (1), rev_2025q1 (1), rev_2025q2 (1), rev_2025q3 (1), rev_2026q1 (1), rev_2026q2 (1), seg_cloud_h1_2026 (1), seg_sub_h1_2026 (1); 2026-10-31: rr_2026_10_sacra70 (2)                                                              | 2026-10-31         | True             |          11 |
| rr_2026_09_s1a | 2026-11-09 | 2026-11-10         | 2026-09-30: rev_2026q1_fortune (1); 2026-10-26: rev_2025fy (1), rev_2025q1 (1), rev_2025q2 (1), rev_2025q3 (1), rev_2026q1 (1), rev_2026q2 (1), seg_cloud_h1_2026 (1), seg_sub_h1_2026 (1); 2026-10-31: rr_2026_10_sacra70 (2); 2026-11-02: rev_2026q3 (1); 2026-11-03: rr_2026_10_semi (2) | 2026-11-03         | True             |          11 |
| rr_2026_10_s1a | 2026-11-09 | 2026-11-10         | 2026-09-30: rev_2026q1_fortune (1); 2026-10-26: rev_2025fy (1), rev_2025q1 (1), rev_2025q2 (1), rev_2025q3 (1), rev_2026q1 (1), rev_2026q2 (1), seg_cloud_h1_2026 (1), seg_sub_h1_2026 (1); 2026-10-31: rr_2026_10_sacra70 (2); 2026-11-02: rev_2026q3 (1); 2026-11-03: rr_2026_10_semi (2) | 2026-11-03         | True             |          11 |

The two seg_* share rows enter B, E, G, R and bayes only (model A's frame has no share rows). Same-day rule: under period_end the Fortune row (published 2026-09-30) enters forecasts dated 2026-09-30; the package's convention is published <= vintage for every row, filing or not.

2. rr_2026_09_s1a and rr_2026_10_s1a (both S-1/A No. 2, published 2026-11-10): entries of one in the other's evidence across all pub files: 0. Their latest vintage is 2026-11-09 (publication rule), and the recorder in forecast_pubfilings.py asserts on every forecast that no evidence row was published on or after the target's own publication date.

3. Answer independence: target value x3, same call as run_backtest (refit, profiled crossing, iv115, pub).

| model   | target         | rule        | vintage    |    mu |     sd |   mu_x3 |   sd_x3 |   abs_diff_mu |   abs_diff_sd |   rel_diff_vs_csv |
|:--------|:---------------|:------------|:-----------|------:|-------:|--------:|--------:|--------------:|--------------:|------------------:|
| A       | rr_2026_10_s1a | publication | 2026-11-09 |  4.33 | 0.0921 |    4.33 |  0.0921 |             0 |             0 |                 0 |
| A       | rr_2026_10_s1a | period_end  | 2026-10-31 |  4.34 | 0.163  |    4.34 |  0.163  |             0 |             0 |                 0 |
| A       | rev_2026q3     | publication | 2026-11-01 | 17.8  | 1.48   |   17.8  |  1.48   |             0 |             0 |                 0 |
| A       | rev_2026q3     | period_end  | 2026-09-30 | 21    | 3.68   |   21    |  3.68   |             0 |             0 |                 0 |
| B       | rr_2026_10_s1a | publication | 2026-11-09 |  4.51 | 0.0701 |    4.51 |  0.0701 |             0 |             0 |                 0 |
| B       | rr_2026_10_s1a | period_end  | 2026-10-31 |  4.75 | 0.119  |    4.75 |  0.119  |             0 |             0 |                 0 |
| B       | rev_2026q3     | publication | 2026-11-01 | 20.1  | 1.32   |   20.1  |  1.32   |             0 |             0 |                 0 |
| B       | rev_2026q3     | period_end  | 2026-09-30 | 23    | 2.36   |   23    |  2.36   |             0 |             0 |                 0 |
