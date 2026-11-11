# Leak audit

Filings are inputs: False. First filing number public: 2026-09-30.

## 1. Hard checks (32 target-vintage pairs): no leaks

| rule        | target         | vintage    |   n_claims |   n_indicator_weeks | latest_published   |   filing_rows |   indicator_ref_after_published | status   |
|:------------|:---------------|:-----------|-----------:|--------------------:|:-------------------|--------------:|--------------------------------:|:---------|
| publication | rr_2025_01     | 2025-09-01 |          1 |                  34 | 2025-09-01         |             0 |                               0 | ok       |
| publication | rev_2025q1     | 2026-10-25 |         16 |                 119 | 2026-10-19         |             0 |                               0 | ok       |
| publication | rev_2025q2     | 2026-10-25 |         16 |                 119 | 2026-10-19         |             0 |                               0 | ok       |
| publication | rev_2025q3     | 2026-10-25 |         16 |                 119 | 2026-10-19         |             0 |                               0 | ok       |
| publication | rr_2025_10     | 2025-10-15 |          4 |                  40 | 2025-10-13         |             0 |                               0 | ok       |
| publication | rr_2025_12     | 2026-03-02 |          8 |                  58 | 2026-03-02         |             0 |                               0 | ok       |
| publication | rev_2025fy     | 2026-10-25 |         16 |                 119 | 2026-10-19         |             0 |                               0 | ok       |
| publication | rr_2026_02     | 2026-02-11 |          6 |                  55 | 2026-02-09         |             0 |                               0 | ok       |
| publication | rr_2026_03     | 2026-03-02 |          8 |                  58 | 2026-03-02         |             0 |                               0 | ok       |
| publication | rr_2026_04     | 2026-04-05 |         10 |                  62 | 2026-03-30         |             0 |                               0 | ok       |
| publication | rev_2026q1     | 2026-10-25 |         16 |                 119 | 2026-10-19         |             0 |                               0 | ok       |
| publication | rr_2026_05     | 2026-05-27 |         11 |                  77 | 2026-05-25         |             0 |                               0 | ok       |
| publication | rev_2026q2     | 2026-10-25 |         16 |                 119 | 2026-10-19         |             0 |                               0 | ok       |
| publication | rr_2026_09_s1a | 2026-11-09 |         17 |                 125 | 2026-11-09         |             0 |                               0 | ok       |
| publication | rev_2026q3     | 2026-11-01 |         16 |                 121 | 2026-10-26         |             0 |                               0 | ok       |
| publication | rr_2026_10_s1a | 2026-11-09 |         17 |                 125 | 2026-11-09         |             0 |                               0 | ok       |
| period_end  | rr_2025_01     | 2025-01-16 |          0 |                   1 | 2025-01-13         |             0 |                               0 | ok       |
| period_end  | rev_2025q1     | 2025-03-31 |          0 |                  12 | 2025-03-31         |             0 |                               0 | ok       |
| period_end  | rev_2025q2     | 2025-06-30 |          1 |                  25 | 2025-06-30         |             0 |                               0 | ok       |
| period_end  | rev_2025q3     | 2025-09-30 |          4 |                  38 | 2025-09-29         |             0 |                               0 | ok       |
| period_end  | rr_2025_10     | 2025-10-15 |          4 |                  40 | 2025-10-13         |             0 |                               0 | ok       |
| period_end  | rr_2025_12     | 2025-12-28 |          6 |                  50 | 2025-12-22         |             0 |                               0 | ok       |
| period_end  | rev_2025fy     | 2025-12-31 |          6 |                  50 | 2025-12-22         |             0 |                               0 | ok       |
| period_end  | rr_2026_02     | 2026-02-09 |          6 |                  55 | 2026-02-09         |             0 |                               0 | ok       |
| period_end  | rr_2026_03     | 2026-02-22 |          8 |                  56 | 2026-02-16         |             0 |                               0 | ok       |
| period_end  | rr_2026_04     | 2026-03-20 |         10 |                  60 | 2026-03-16         |             0 |                               0 | ok       |
| period_end  | rev_2026q1     | 2026-03-31 |         10 |                  62 | 2026-03-30         |             0 |                               0 | ok       |
| period_end  | rr_2026_05     | 2026-05-15 |         11 |                  73 | 2026-05-11         |             0 |                               0 | ok       |
| period_end  | rev_2026q2     | 2026-06-30 |         13 |                  87 | 2026-06-30         |             0 |                               0 | ok       |
| period_end  | rr_2026_09_s1a | 2026-09-30 |         16 |                 113 | 2026-09-28         |             0 |                               0 | ok       |
| period_end  | rev_2026q3     | 2026-09-30 |         16 |                 113 | 2026-09-28         |             0 |                               0 | ok       |
| period_end  | rr_2026_10_s1a | 2026-10-31 |         16 |                 121 | 2026-10-26         |             0 |                               0 | ok       |

## 2. Rows published after the first filing number

| obs_id             | origin_id            | tier        |   value | kind   | published           | risk                | notes                                                                                                                                                                                                                                                                            |
|:-------------------|:---------------------|:------------|--------:|:-------|:--------------------|:--------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| fc_nyt_2026_09     | nyt_2026-09_forecast | leak_tier1  |     100 | lower  | 2026-09-30 00:00:00 | independent source  | original forecast wording; the 'already passed $100B' re-reports are excluded as a misquote                                                                                                                                                                                      |
| rr_2026_10_sacra70 | sacra_2026-10_70b    | third_party |      70 | point  | 2026-10-31 00:00:00 | excluded (filing=2) | CONFLICT: Sacra's company page says $80B in October (see rr_2026_10_sacra80). Dates approximate; filing=2: published after the public S-1 (2026-10-26); may be computed from it                                                                                                  |
| rr_2026_10_semi    | semi_2026-10_74b     | third_party |      74 | point  | 2026-11-03 00:00:00 | excluded (filing=2) | provenance unclear: described both as 'mid-October estimate' and 'annualized the new quarterly data'; if derived from Q3 GAAP it double-counts. Published date conservative; filing=2: published after S-1/A No.1 (Q3 revenue); described as 'annualized the new quarterly data' |
| rr_2026_10_wsj     | wsj_2026-11-06_80b   | leak_single |      80 | lower  | 2026-11-06 00:00:00 | independent source  | single anonymous source -> low weight; 'topped' -> lower bound                                                                                                                                                                                                                   |
| rr_2026_10_sacra80 | wsj_2026-11-06_80b   | third_party |      80 | lower  | 2026-11-07 00:00:00 | excluded (filing=2) | appears to echo WSJ -> same origin, removed by dedup; filing=2: published after the public S-1; echoes WSJ                                                                                                                                                                       |

## 3. Breaks: when they became detectable

| indicator      | break_date   | first_detectable   |   days_after |
|:---------------|:-------------|:-------------------|-------------:|
| pypi_anthropic | 2026-08-25   | 2026-09-12         |           18 |

Targets whose vintage falls between a break and its first detectable day:

none

## 4. Parameters

| pipeline                          | parameters                                                                         | estimated from                                                                              | status                                             |
|:----------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|:---------------------------------------------------|
| State-space backtest, refit=True  | process noise (sigma_l, sigma_g, sigma_b, kappa)                                   | refit on data public at the vintage                                                         | ok                                                 |
| State-space backtest, refit=True  | indicator beta / drift / obs_sd                                                    | reset to catalog priors, refit at the vintage                                               | ok                                                 |
| State-space backtest, refit=True  | crossing position rho                                                              | profiled at the vintage with unfitted parameters (before this audit: with full-sample fits) | fixed                                              |
| State-space backtest, refit=False | all of params.json                                                                 | full-sample fits                                                                            | LEAKS: not used for forecast results               |
| Bayesian backtest                 | about 20 parameters                                                                | MAP + Laplace at the vintage from priors; warm starts now ordered by vintage                | ok                                                 |
| ML backtest                       | smoother labels                                                                    | process noise refit at the vintage (before this audit: full-sample params.json)             | fixed                                              |
| all                               | source noise by tier, growth_longrun, definition lag, catalog priors, holiday rule | set by hand with the whole history in view                                                  | hindsight, see audit.py sensitivity                |
| Price-adjusted indicators         | model mix weights                                                                  | uninformed 50/50 (ok); measured GitHub mix up to the date (ok); Nov 2026 mix                | nov2026_mix is HINDSIGHT                           |
| Price-adjusted indicators         | gamma (price elasticity)                                                           | grid; real-time likelihood weights in pricing.gamma_weights                                 | picking gamma = 1 after seeing errors is HINDSIGHT |

## 5. Alternative-data vintages

Values are recorded with the date they were fetched or published. A snapshot fetched today cannot enter a backtest before today; daily download counts are not revised.

| indicator                  | first_date   | last_date   | published_min   | published_max   |   rows |   published_before_date |
|:---------------------------|:-------------|:------------|:----------------|:----------------|-------:|------------------------:|
| npm_anthropic_sdk          | 2025-01-01   | 2026-11-09  | NaT             | NaT             |    678 |                       0 |
| npm_claude_agent_sdk       | 2025-01-01   | 2026-11-09  | NaT             | NaT             |    678 |                       0 |
| npm_claude_code            | 2025-01-01   | 2026-11-09  | NaT             | NaT             |    678 |                       0 |
| npm_openai                 | 2025-01-01   | 2026-11-09  | NaT             | NaT             |    678 |                       0 |
| npm_openai_codex           | 2025-01-01   | 2026-11-09  | NaT             | NaT             |    678 |                       0 |
| pypi_anthropic             | 2026-04-08   | 2026-11-09  | NaT             | NaT             |    216 |                       0 |
| pypi_openai                | 2026-04-08   | 2026-11-09  | NaT             | NaT             |    216 |                       0 |
| ramp_adoption_anthropic    | 2026-08-31   | 2026-09-30  | 2026-11-10      | 2026-11-10      |      2 |                       0 |
| ramp_token_spend_anthropic | 2026-09-25   | 2026-10-23  | 2026-11-10      | 2026-11-10      |      2 |                       0 |

## 6. Choices made after seeing results (not testable on the same targets)

- Which indicators have role=model (npm, PyPI, Ramp) was decided after the layer backtest on the same targets.
- The crossing-position model was added after seeing errors on crossing claims; rho itself is estimated per vintage.
- Catalog priors (beta 0.75, drift, obs_sd), tier noise and the holiday window were chosen with the full history visible.
- The run-rate definition (t28x13, 14-day lag) became public around May 2026; earlier vintages use it with hindsight (the Bayesian model mixes 7 and 14 days).
- Price adjustment with gamma = 1 has the lowest error after the fact, but at every vintage the likelihood then available puts about 70-90% of the weight on gamma = 0. Reporting gamma = 1 as the model would be a hindsight choice.
- GitHub commit counts were all measured on 2026-11-11. Counts for earlier weeks include commits that became public later (repositories opened later, late pushes), so the backfilled series is not what a forecaster would have seen at the time.
- Each of these makes the backtest look better than a true real-time forecast would have. Only targets that arrive after a forecast is registered (forecasts/registry.jsonl) are free of it.
