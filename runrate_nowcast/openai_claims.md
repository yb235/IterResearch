# OpenAI revenue claims, 2024-01 to 2026-11 (step 2)

Data: [`data/openai_claims.csv`](data/openai_claims.csv), 56 rows. It has the same columns as `observations.csv` plus `company = openai`, and it passes `data._validate`.

- **Observations:** 37 rows. 35 are included. The two Axios/Reuters "$70B" rows are excluded (see conflicts).
- **Forecasts and targets:** 19 rows (`is_forecast = 1`).
- **Duplicates:** 4 `origin_id`s carry a second report of the same figure. After dedup there are 32 included observations.
- **Verification:** every row comes from search summaries, and each `notes` field says `search summary`. WebFetch could not reach any host tried (investinglive, finance.yahoo, cryptobriefing, thestar, epoch.ai), so no original article was read. Most URLs point to syndicated copies.
- **Conservative published dates:** three rows use a later date than the true one, because the exact date was not seen: the Feb-2026 The Information projections (set to Feb 28), the FT "nearly $2B/month" figure (Jun 30) and the Epoch AI compilation (Oct 14, 2025).

## Main series (OpenAI's own basis)

| Ref date | Value | Kind | Tier | Source (published) |
|---|---|---|---|---|
| Dec 2023 | $1.6B / $2.0B | run-rate | leak_tier1 | The Information (12-30) / FT (2024-02-09); conflict |
| early Jun 2024 | $3.4B | run-rate | leak_single | The Information (06-12), Altman to staff |
| Aug 2024 | $0.3B | monthly revenue | leak_tier1 | NYT documents (09-27) |
| FY2024 | $3.7B | revenue | leak_single, later audited | Bloomberg (2025-03-26); leaked audit (2026-06-15) |
| end 2024 | $5.5B / $6B | run-rate | official | CNBC (2025-06-09) / Friar blog (2026-01-18) |
| early Jun 2025 | $10B | ARR | official | spokesperson to CNBC (06-09) |
| H1 2025 | $4.3B | revenue | leak_tier1 | The Information (09-29) |
| Jul 2025 | $12B; July revenue ≥ $1B | run-rate; monthly revenue | leak_single; official | The Information (07-30); Friar on CNBC (08-20) |
| Aug 2025 | $13B | run-rate | secondary | Epoch AI; original source not found |
| end 2025 | $21.4B (and "$20B+") | run-rate | leak_tier1 (official) | The Information (2026-03-04); Friar (01-18) |
| FY2025 | $13.1B / $13.07B | revenue | leak_tier1 | CNBC (2026-02-20); leaked audited statements (06-15) |
| end Feb 2026 | > $25B | run-rate | leak_tier1 | The Information (03-04) |
| Mar 2026 | $2B per month (≈ $24B) | official | official | funding-close release (03-31) |
| early Apr 2026 | ~$25B | run-rate | leak_single | Semafor (04-08) |
| Q1 2026 | $5.7B | revenue | leak_tier1 | The Information (05-21, 06-16) |
| Q2 2026 | $6.7B | revenue | leak_tier1 | WSJ (08-18) |
| end Jun 2026 | ~$28B | **derived** | leak_tier1 | $50B ÷ 1.77 (Q3 growth in the investor deck) |
| early Aug 2026 | passed $40B | run-rate | leak_tier1 | Bloomberg (08-13); July +20% month over month (Brockman memo) |
| end Sep 2026 | ~$50B | run-rate | leak_tier1 | FT / The Information / CNBC (10-08) |
| Q3 2026 | ~$10B | revenue | leak_tier1 | The Information and Bloomberg (11-10) |

Targets, in date order: FY24 $3.7B and FY25 $11.6B (NYT, 2024-09); FY25 $11B (Friar, 2025-02); FY25 $12.7B and FY26 $29.4B (Bloomberg, 2025-03); FY25 $13B (The Information, 2025-09); run-rate above $20B at end 2025 (Altman, 2025-11-06); FY26 $30B and FY27 $62B (The Information, 2026-02); FY26 $36B (FT, 2026-09-18); run-rate of $70B or more at end 2026 (Bloomberg, 2026-10-09). There are also 2029 and 2030 targets (NYT, The Information, CNBC, FT).

## Coverage by month (observations, by reference month)

- **2023-12:** 4 rows (anchors).
- **2024:** Jun 1, Aug 1, Dec 5 (two run-rates and three FY revenue rows).
- **2025:** Jun 2, Jul 2, Aug 1, Sep 1 (a derived lower bound on Q3 2025), Oct 1, Dec 6.
- **2026:** Feb 1, Mar 3, Apr 1, Jun 2, Aug 1, Sep 5 (two of them excluded).

The densest periods are:

- **Dec 2025 to Apr 2026:** year-end figures, then the $25B reports and the official $2B per month.
- **Aug to Nov 2026:** the $40B report, Q2, the $70B vs $50B dispute, the year-end target, Brockman's on-record growth figure, and Q3.

**Months in 2026 with no run-rate observation:** January, May, July (only a growth rate), October and November. June has only the derived value. The only 2026 revenue data are quarterly (Q1–Q3); there are no monthly revenue figures.

The 2024 data are thin: three run-rate points across the whole year.

## Conflicts

- **Dec 2023:** $1.6B (The Information) vs $2B (FT, a month's revenue × 12).
- **End 2024:** the company gave $5.5B to CNBC, and Friar's blog later said $6B.
- **2025 target:** $10B ("target", CNBC 2026-02), but projections at the time ranged from $11B to $13B (NYT, Friar, Bloomberg, The Information).
- **Altman's "well more than $13B" (Nov 2025):** it is ambiguous whether he meant revenue or run-rate. It is entered as a weak run-rate lower bound.
- **Mar 2026:** the official "$2B a month" works out to about $24B, slightly below The Information's "> $25B" at the end of February. The likely cause is rounding, or a run-rate that stayed flat; Semafor and the WSJ reported that OpenAI missed monthly targets in early 2026.
- **Sep 2026, $70B (Axios, Reuters) vs $50B (FT, The Information, CNBC):** per the FT, investors computed the $70B on Anthropic's method, grossing up cloud-partner sales, and a source told CNN the figure did not come from OpenAI. Both $70B rows have `include = 0`.
  - On that gross basis the ratio is about 1.4×.
  - Axios's "+70% since the start of Q3" implies about $41B at the start of Q3 on the gross basis, against about $28B on OpenAI's own basis.
- **Bloomberg Aug 13:** reports differ on the wording, "has surpassed $40B" vs "set to top $40B".
  - The $40B fits OpenAI's net basis: a path from about $28B (end of June) to about $50B (end of September) averages about $40B, which matches Q3 revenue of about $10B.
- **Q3 growth:** 77% in the investor deck (CNBC) vs Brockman's on-record 70%. The implied end-June level is $28.2B or $29.4B.

## Definitions: OpenAI vs Anthropic

**OpenAI** calls its figure "ARR" or "annualized revenue". The sources describe it in three ways:

- **Four weeks × 12** (The Information, 2026-03): OpenAI multiplies its last four weeks of revenue by 12. That is 336/365 ≈ 0.92 of the true annual pace.
- **A month × 12** (FT, 2024): an earlier description of the same idea.
- **What is in it** (CNBC, 2025-06): consumer, business and API revenue. It excludes Microsoft licensing revenue and large one-time deals.

There is a cross-check: $21.4B at the end of 2025 is $1.78B per four weeks, which is about $1.97B for a 31-day December. That matches the FT's "nearly $2B a month".

OpenAI also counts **net** revenue. Sales of OpenAI models through Azure are not OpenAI revenue; only its revenue share is (2025 revenue includes about $0.3B from Microsoft).

**Anthropic** uses `t28x13`: trailing 28 days of consumption × 13, plus subscriptions × 12. That is roughly a true annualization. It also books cloud-marketplace sales **gross**, about 48% of H1 2026 revenue.

**Consequences for the model:**

- For the same 28 days of consumption revenue, OpenAI's number is 12/13 ≈ 0.92 of Anthropic's. Subscriptions are × 12 on both sides.
- The gross vs net channel accounting is the larger gap: about 1.4× at the end of September 2026. Fit the two companies with separate level and offset terms.
- Rows whose source states the computation are tagged `month_x12` (15-day lag; four weeks would be about 14). All other rows are `unknown`.
- As a rough check, quarterly revenue × 4 is within about 3% of the average of the start- and end-of-quarter run-rates for Q1–Q3 2026 (22.8 vs 23.2, 26.8 vs 26.6, 40 vs 39.1). This suggests the run-rate and recognized revenue are on the same net basis. The 4wk × 12 shortfall (−8%) does not show up in this check, which is too coarse given the rounding.

**Not in the CSV.** The segment codes are Anthropic-specific, so these OpenAI segment figures were left out:

- Enterprise share: over 40% of revenue (Mar 2026), a majority by Q3 2026 (Brockman).
- Ads: $1B annualized (OpenAI, 2026-08-31).
