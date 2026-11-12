# Consumer signals from published reports (collected 2026-11-12)

Data: [`data/consumer_signals.csv`](data/consumer_signals.csv), 175 rows (104 Claude, 49 ChatGPT, 22 Gemini). Each row is one figure as an article reported it. Nothing is interpolated, and conflicting figures are kept as separate rows. The `published` column holds the earliest date found for the article or post. When only a repost or mirror was dated, that date is used as an upper bound and the notes say so; using a later date is conservative for leakage. 69 rows have no date.

**How the figures were collected.** Every vendor and news host was blocked from this environment: WebFetch could not resolve appfigures.com or officechai.com, and curl to Similarweb, Sensor Tower, AppMagic, FourWeekMBA, ppc.land, Forbes and TechCrunch timed out. So every row is `verification = search summary`: a search engine's summary of the article, not the page itself. Spot-check the values before relying on them, especially rows whose notes contain LOW CONFIDENCE, SUSPECT or CONFLICT.

## Rows by series

| Series | Rows | Content |
|---|---|---|
| `web_visits_monthly` | 40 | Monthly visits for claude.ai (22 rows, including quarterly totals and 12-month averages), chatgpt.com (14) and gemini.google.com (4). All Similarweb except one Semrush row and one AICPB row |
| `web_traffic_share` (+`_us`) | 40 | Similarweb share of generative-AI website visits, worldwide (US for 2026-10 only) |
| `web_visits_daily*`, `_28d`, `_weekly`, `_growth` | 10 | Daily averages, record days, the 28-day rolling total and MoM growth |
| `web_visits_claude_code*`, `web_unique_visitors*` | 10 | claude.ai/code visits and users; claude.ai unique visitors |
| `web_rank_global`, `comscore_us_desktop_uv`, `sessions_*` | 9 | Similarweb global rank; Comscore; Graphite web+app sessions |
| `app_downloads_*`, `app_download_share` | 24 | Appfigures, AppMagic, Apptopia (US) and Sensor Tower |
| `app_revenue_*`, `app_arpu_us` | 25 | Gross consumer spend vs net revenue (marked in `metric`/`unit`), revenue per download, US ARPU |
| `app_users*`, `app_dau_share_us` | 13 | Sensor Tower MAU and shares; Apptopia US DAU share |
| `app_store_rank_us`, `paid_subscribers` | 4 | US App Store rank; Anthropic's "more than doubled"; Indagari +75% |

## Monthly coverage (Claude unless stated)

Units: web visits in millions per month; share in %; app downloads in millions (worldwide); revenue in $M. `pub` = earliest publication date found.

| Month | claude.ai visits (pub) | SW share | chatgpt.com visits (B) | Gemini visits (B) | App downloads | App revenue |
|---|---|---|---|---|---|---|
| 2025-01 | – (Q1 total ~230) | – | 3.8 | – | – | – |
| 2025-02 | – | – | – | – | – | – |
| 2025-03 | 3.3M/day avg (04-01) | – | – | 10.9M/day | – | – |
| 2025-04 | – | – | 4.79 | – | – | – |
| 2025-05 | ~100 (2025-06-11); 99.7 | 1.6* | 5.5 | – | – | – |
| 2025-06, 07 | – | – | – | – | ~1.2 "typical 2025"* | – |
| 2025-08 | 148.49 (2025-09-10) | 1.9* | 5.85 | – | – | – |
| 2025-09 | – | – | – | – | – | – |
| 2025-10 | 157 (Neil Patel, undated) | 2.3* | 6.2* | – | – | ChatGPT 187 net |
| 2025-11 | – | – | – | – | – | – |
| 2025-12 | 176 (AICPB, not SW) | 2.22 | – | – | – | – |
| 2026-01 | 203 (+17.5% MoM) | 2.1 | 5.72 | 2.07 | – | 7.5 gross (AppMagic, 08-28) |
| 2026-02 | 203 per SW blog (CONFLICT); +43% MoM (03-06); 11M/day | ≥3 | 5.35 | 2.11 | ~5 (pacing) | ~11 net (pacing) |
| 2026-03 | 614; 20M/day | 6.02 (04-16) | 5.73 | – | 6.6 (low conf.) | App Store +148% MoM |
| 2026-04 | 824 SW / 838 Semrush | 8.0* | 5.51 | 2.76 | 20 AF; 27 AppMagic? | 76 gross (AF) |
| 2026-05 | 952.6 (~06-10/12) | 8.9 (06-11) | 5.6 | – | 18 AF (06-04); 29 AppMagic? | 92 gross (AF, 06-04) |
| 2026-06 | ~947 (09-17) | 9.2 (07-15) | 5.4 | – | – | – |
| 2026-07 | 969 (09-17) | 9.6* / 9.8* | – | – | 16.6 AppMagic (08-28) | 97.2 gross AppMagic (08-28) |
| 2026-08 | 950 (09-17) | 9.3 (~09-07) | 5.6 | – | US only: 3.2 (Apptopia) | 82 **net** AF (09-18) |
| 2026-09 | 996.4 (10-03) | 9.6 (10-12) | 5.9 (projection) | – | – | – |
| 2026-10 | 1,250 (11-05); 28-day 1,101 to 10-22 (10-28) | 11.7 / 11.5 (11-05) | 6.05 (11-08) | 2.64 | – | – |

\* Known only from a later article (a year-ago comparison or a retrospective series), so not available in real time.

**Missing months.**
- claude.ai monthly visits: 2025-01 to 04 (only the Q1 total and a March daily average), 2025-06, 07, 09 and 11.
- Similarweb share: most of 2025.
- Gemini visits: 2025 and 2026-05 to 09.
- ChatGPT visits: 2026-07.
- Claude app downloads (worldwide): every 2025 month, 2026-01, 06, 09 and 10. August exists for the US only.
- Claude app revenue: every 2025 month, 2026-03, 06, 09 and 10.

No article reported Google Trends index values for Claude with a date, so there is no Trends series. App Store rank has only two points: #124 on 2026-01-28 and #1 on 2026-02-28 or 03-01.

## How consistent the vendors are

- **Web, Similarweb vs Semrush:** April 2026 is 824 vs 837.7 (+1.7%). Some aggregators credit May's 952.6 to Semrush.
- **Web, Similarweb's own labelling:** Similarweb's IPO blog dates the 203M month to **February**, while ppc.land (citing Similarweb) dates it to January. January fits the other figures better: Q1 2026 ≈ 1.08B, Feb +43% MoM and March 614M give 203 + ~290 + 614 ≈ 1.1B.
- **Web, growth rates:** BofA's Feb +63% (daily average) and aigazine's +43% (monthly total) disagree.
- **Share:** October is 11.5 (OfficeChai) vs 11.7 (Momentic); July is 9.6 vs 9.8. Cryptobriefing's "8.0% in March" is Similarweb's April figure; March was 6.02. Similarweb's "monthly updates" are sometimes measured to the 26th rather than to month end.
- **App downloads:** Appfigures (Apr 20M, May 18M) and AppMagic (July 16.6M, about −3.5% vs June) agree on level. A Mobile Time line credits AppMagic with 27M → 29M for Apr → May, which contradicts both and is kept only as low confidence. For ChatGPT, AppMagic (77.8M in July) runs well above Appfigures (51–61M in H1), so **do not splice vendors**.
- **App revenue:** gross and net are mixed. Appfigures gives May $92M gross and Aug $82M net; AppMagic gives July $97.2M gross. Store fees are 15–30%, so $82M net is roughly $96–117M gross: the same trend, but the rows are not comparable without an assumption. Appfigures' Jan–Apr 2026 cumulative $156M gross is consistent with AppMagic's January $7.5M and the $76M April figure.
- **Users:** Sensor Tower's 245M "AI assistant" MAU (May) and its 56M app MAU (Q2) use different definitions. Graphite's web+app session shares (Claude 7.2%, Gemini 4.9% in September) do not fit Similarweb's visit shares (9.6% and 25.0%), so treat them as a separate universe.

## Which series could feed a June–October 2026 monthly feature

| Series | Jun | Jul | Aug | Sep | Oct | First-release lag after month end | Verdict |
|---|---|---|---|---|---|---|---|
| claude.ai visits (Similarweb) | ~947 | 969 | 950 | 996.4 | 1,250 | Now 3–17 days (Sep: Forbes 3d; Oct: Momentic 5d; Aug: primer 17d). Jun and Jul surfaced only in the 09-17 primer (79d and 48d) | **Usable** from August on, with a ~1–2 week lag. In a backtest, use the published dates, not month end |
| Similarweb share, Claude | 9.2 | (9.6, retro) | 9.3 | 9.6 | 11.7 | 5–15 days (X/LinkedIn posts ~day 5–12; news 0–3 days later) | **Usable**; the most regular series. July was only published in retrospect |
| claude.ai 28-day rolling / daily record | | | | | 1,101 | 6 days after window end | Occasional; good for nowcasting the current month |
| chatgpt.com / Gemini visits | 5.4 / – | – / – | 5.6 / – | 5.9p / – | 6.05 / 2.64 | 3–8 days | Not complete; use the share series for comparisons |
| Claude app revenue | – | 97.2 g | 82 n | – | – | 18–28 days | **Not usable** as a monthly series: gaps plus mixed vendors and gross/net |
| Claude app downloads | – | 16.6 | (US 3.2) | – | – | 28 days | Not usable |
| claude.ai/code visits | – | – | 8.5 | 11.2 | – | 17–35 days | Too short; worth tracking from Similarweb's chart posts |
| Sensor Tower quarterly | Q2: Claude #2 GenAI revenue | | | Q3: MAU +1,260% YoY | | ~6 weeks (Q2 data → 08-11) | Quarterly cross-check only |

Observed cadence: Similarweb posts monthly shares and the top-websites ranking on X/LinkedIn about 5–12 days after month end, and Forbes, FourWeekMBA, OfficeChai and Momentic repeat them within 0–3 days. Appfigures posts monthly top-app insights 4–18 days after month end. AppMagic numbers reached news about 4 weeks after month end (one exclusive). Comscore takes about 7–8 weeks.

Reading: claude.ai web visits rose about 4x from January to May 2026, were **flat May–August** (947–969M), and then rose 5% in September and 24% in October. Mobile revenue kept growing through July and August. Neither tracks revenue in level, so these should enter the model as growth rates with drift, like the other indicators.

## Data-quality concerns

1. Every value comes from a search summary. Summaries sometimes shift months or years. Examples:
   - aigazine's weekly series is dated Dec 2024–Feb 2025 at levels that only fit Dec 2025–Feb 2026.
   - OfficeChai's October piece gives ChatGPT 69.5% and Gemini 15.9%. These are not recorded.
2. Several 2025 values are known only as year-ago comparisons published in 2026. Treat them as public on the later date. The exception is Aug 2025 (148.49M), which Siècle Digital reported on 2025-09-10.
3. Problems in existing files (left unedited):
   - `data/disclosures.csv` cites the Forbes AU article (about June 2026) for Aug 2025 148M, Jan 2026 203M, Apr 824M and Aug 2026 950M. Aug 2026 cannot come from a June article. These figures appear in Similarweb's pre-IPO primer (2026-09-17, via ppc.land) and its IPO blog, so their `published` dates are later than the Forbes date implies.
   - The disclosures row for May 2026 (952.6M) is fine.
4. Projections are kept as reported and marked "pacing" or "projection" in `unit`: Appfigures' February pacing figures and Forbes' September projection for ChatGPT.
5. Undated rows (69) are mostly OfficeChai, aigazine, Apptopia and Appfigures monthly posts. Their typical lags above can serve as a fallback, but they are not recorded as dates.
