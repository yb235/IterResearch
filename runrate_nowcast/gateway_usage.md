# Second usage panel: model gateways and coding-agent platforms (task D, 2026-11-12)

**Bottom line.** Vercel AI Gateway is the only source that can give a second panel for June to October 2026. Two parts of it matter:

- **Now:** Vercel's monthly *AI Gateway Production Index* gives Anthropic's spend share for April through October, and each edition has a publication date.
- **Once `vercel.com` is allowed:** its public export endpoint gives the daily lab shares of tokens and spend for any date range.

Use the **spend share**, not the token share. Since June, cheap open-weight models have flooded the token counts.

The other platforms do not help:

- Requesty, Portkey, Kilo and OpenCode have no Anthropic figures for June to October that we can reach, or their figures are undated.
- Helicone, LiteLLM, Martian, Cloudflare and Vantage publish nothing usable.

The snapshots are in [`data/gateway_usage_snapshots.csv`](data/gateway_usage_snapshots.csv): 64 rows, with shares in percent (0–100). Every row is `verification = search summary`. No page could be read directly, because every target host fails DNS here: vercel.com, requesty.ai, blog.pebblous.ai, token-race.vercel.app, and the others listed in section 5.

## 1. Vercel AI Gateway, monthly Production Index

Each edition covers the previous month. "Published" is the edition date; where it was taken from an X post ID or inferred, the CSV says so.

| Data month | Published | Anthropic token share | Anthropic spend share | Open-weight token / spend share | Other |
|---|---|---|---|---|---|
| Apr 2026 | 05-13 | 26 | 61 | ~11–13 / – | Google 38 tok / 21 spend; OpenAI 13 / 12 |
| May | 06-09 | 32 | 65 | – | DeepSeek 17 tok / ~1 spend |
| Jun | 07-13 | 32 | 61 | 29 / <4 | DeepSeek 22.6 tok; Google 24 tok (may be May) |
| Jul | 08-11 | 29.8 | 65.1 | 36 / – | Gateway spend +37% m/m, tokens +59%; DeepSeek ~25 tok, OpenAI 12.8, Google ~11 |
| Aug | ~09-17 | – | 64 (revised to 63.7 on 10-15) | 56 / 14 | Opus 5 = 22.5% of spend; price per token −23% |
| Sep | 10-15 | ~11 (Cache Register, 4-week average at end of Sep) | 43.9 (Cache Register: 47) | 71.5 / 23 | DeepSeek 58 tok / 6.6 spend; first month with Anthropic under 50% of spend |
| Oct | 11-09 | – | ~32 (**unverified**, conflicting summaries) | – | Opus 5.5 is the top model at ~14% of spend; new teams choose Anthropic 2:1; OpenAI reaches 55% of paid teams, Anthropic 54% |

Coverage also says Anthropic held at least 61% of spend every month from December 2025 to August 2026. Total gateway spend in September was about 13 times December's, while the average price per token fell about 47%.

**Does it show the Q3 dip and the October rebound?** Only in part.

- There is **no June–August dip** in Anthropic's spend share (61 → 65 → 64). Gateway spend was growing fast (+37% in July), so Anthropic's absolute Vercel spend probably still rose through August.
- There is a **sharp break in September** (64 → 44). The timing matches the OpenRouter trough in mid-September. Absolute Anthropic spend on Vercel fell in September unless total gateway spend grew by more than about 45% month on month; Vercel did not publish that growth figure.
- There is **no October rebound in share**. Vercel says the share fell again in October, even though Opus 5.5 became the top model by spend.
- The September–October fall is partly mechanical. Spend is valued at list prices, which were cut on Sep 22 (Opus 5.5, −20%) and Oct 7 (Haiku 5.5, −90%). This is relevant to revenue, which falls in the same way.
- The token share collapsed from ~30% in July to ~11% at the end of September. That is the open-weight surge, not an Anthropic decline.

A share series cannot show whether Anthropic's level rebounded in October. The export endpoint gives shares only; Vercel never publishes absolute volumes.

The **rolling-window leaderboard copies** under `vercel.com/<team>/<project>/ai-gateway/leaderboards` (18 rows in the CSV, most with windows ending 13–31 July) disagree with each other. For example, the windows May 16–Jul 14 and May 17–Jul 15 show 16.8% and 28.4%. The page has 2W/1M/2M toggles, and the captured number often does not match the displayed dates. Keep these rows as raw snapshots and do not use them as a series. The canonical labs page currently shows the window Aug 13–Nov 10, but no percentages were captured from it.

## 2. Which platforms could give a weekly or monthly series for June–October 2026

| Platform | What is public | Frequency / history | Anthropic values found for Jun–Oct 2026 | Independent of OpenRouter? | Verdict |
|---|---|---|---|---|---|
| **Vercel AI Gateway** | Monthly index; leaderboards; **open export API** (CC BY 4.0) | Daily data, any `from`/`to` range (docs example starts 2025-10-01) | Monthly spend share Jun–Oct (above) | Yes | **Best candidate** |
| Portkey | `portkey.ai/rankings/{provider}/{daily,weekly,monthly}`: requests, tokens and cost per provider and model; "provider market share"; updated daily | Daily / weekly / monthly views; history depth unknown | Anthropic direct API: 19.86T tokens and $18.7M (daily view), 155.9T and $139M (monthly view); no dates | Yes (enterprise traffic) | Second candidate: absolute levels, split by direct API / Bedrock / Vertex |
| Kilo Code | `kilo.ai/leaderboard/race`: weekly tokens per lab, Jun 2025–Oct 2026; `leaderboard/llm.txt` | Weekly | Only the first week (Jun 2025: Anthropic 26.4B, Google 32.6B) | **No**: Kilo is a top OpenRouter app | Robustness check only |
| Requesty | `/data` datasets (coding-agent model share for Apr 2026; OSS share Nov 2025–Apr 2026); `/rankings` (window 2026-06-01 to 08-03) | Ad hoc releases | Rankings window values not captured; one "early August" figure (51% of spend) is unverified | Yes | Not a series; Claude is ~90%+ of coding-agent spend, so little variation |
| OpenCode | `opencode.ai/data`, `stats.opencode.ai` | Daily table, 2-month and 7-day views | Anthropic about **0.1%** of OpenCode tokens | Yes | Useless for Anthropic (see the April restriction below) |
| Cline | Blog post only (Oct 2025, Plan/Act model shares) | One-off | None | Partly via OpenRouter | No |
| Roo Code | No own leaderboard found; Requesty April figure only | – | None | – | No |
| Helicone | `helicone.ai/stats/providers/anthropic` (no numbers in snippets) | ? | None | Yes | Maintenance mode since the Mintlify acquisition (Mar 2026) |
| LLM Gateway | `llmgateway.io/rankings` ("provider market share") | ? | None captured | Yes | Small; check once reachable |
| LiteLLM, Martian, Cloudflare AI Gateway | Nothing public on model share | – | – | – | No |
| Vantage | `routers.vantage.sh` = OpenRouter provider economics; `tokens.vantage.sh` not found | – | – | **No** (OpenRouter-derived) | No |
| Dirac (`dirac.run/labs-market-share`), Cache Register | Compilations | – | Cache Register: Vercel end-Sep 4-week average | Dirac uses OpenRouter | Cross-check only |

## 3. Direct data URLs to fetch once the hosts are allowed

**Vercel (priority).** The endpoint parameters are `dataset` = models | labs | apps | providers, `modality` = all | text | image | video, and `from` / `to`; without `from`, it returns a rolling 2 months. It is cached for 24 hours. Each row is per day with `share_percent`, and the docs list `metric` values of requests, tokens, spend, imageCount and videoCount. It is not confirmed whether `metric` is a query parameter or only a column.

- `https://vercel.com/api/ai/leaderboard-export?dataset=labs&modality=text&from=2025-10-01&to=2026-11-10&format=csv`
- `https://vercel.com/api/ai/leaderboard-export?dataset=models&modality=text&from=2025-10-01&to=2026-11-10&format=csv`
- `https://vercel.com/api/ai/leaderboard-export?dataset=providers&from=2025-10-01&to=2026-11-10&format=csv` (direct Anthropic vs Bedrock vs Vertex)
- The same URLs without `format=csv` return JSON.
- Documentation: `https://vercel.com/docs/ai-gateway/leaderboards.md` and `https://vercel.com/changelog/open-data-and-shareable-charts-for-ai-gateway-leaderboards`
- Leaderboard pages: `https://vercel.com/ai-gateway/leaderboards/labs`, `/models?range=2m`, `/providers`
- Index posts, one per month:
  - `https://vercel.com/blog/ai-gateway-production-index`
  - `https://vercel.com/blog/ai-gateway-production-index-june-2026`
  - `https://vercel.com/blog/ai-gateway-production-index-july-2026`
  - `https://vercel.com/blog/deepseek-overtakes-google-on-volume-cost-per-token-falls`
  - `https://vercel.com/blog/ai-gateway-production-index-september-2026`
  - `https://vercel.com/blog/ai-gateway-production-index-october-2026`
  - `https://vercel.com/blog/ai-gateway-production-index-november-2026`

**Portkey**

- `https://portkey.ai/rankings/monthly` (also `/daily` and `/weekly`)
- `https://portkey.ai/rankings/anthropic/{daily,weekly,monthly}`
- `https://portkey.ai/rankings/bedrock/{daily,weekly}`
- `https://portkey.ai/rankings/vertex-ai/monthly`

**Requesty**

- Indexes: `https://www.requesty.ai/data`, `https://www.requesty.ai/data/llms.txt` and `https://www.requesty.ai/data/llms-full.txt` (these list all datasets, including any later months)
- Coding-agent model share: `https://www.requesty.ai/data/coding-agent-model-share-apr-2026/data.json` (and `data.md`)
- OSS family share: `https://www.requesty.ai/data/oss-family-share-jan-apr-2026/data.json`
- Rankings: `https://www.requesty.ai/rankings` and `https://www.requesty.ai/rankings/data.md`
- `https://www.requesty.ai/papers/coding-agents/The-Coding-Agent-Economy.pdf`

**Kilo**

- `https://kilo.ai/leaderboard/race`
- `https://kilo.ai/leaderboard/llm.txt`
- `https://kilo.ai/leaderboard`
- `https://kilo.ai/models/by/anthropic`

**OpenCode**

- `https://opencode.ai/data/anthropic.md`
- `https://opencode.ai/data/`
- `https://stats.opencode.ai/`

**Others**

- `https://www.helicone.ai/stats/providers/anthropic`
- `https://llmgateway.io/rankings`
- `https://cacheregister.dev/source-matters/`
- `https://blog.pebblous.ai/report/vercel-ai-gateway-token-spend-split-2026-10/en/` (recomputes Vercel's monthly spend shares from the export)

**Historical vintages.** Use the Wayback CDX API at `https://web.archive.org/cdx/search/cdx?url=vercel.com/ai-gateway/leaderboards*&output=json`, and the same pattern for `portkey.ai/rankings*` and `kilo.ai/leaderboard*`.

## 4. Hosts to allow

- **Essential:** `vercel.com` and `web.archive.org`. Both are already on DATA_PLAN's list.
- **Useful:** `portkey.ai`, `kilo.ai`, `requesty.ai` / `www.requesty.ai` (already listed), `opencode.ai`, `stats.opencode.ai`.
- **Optional:** `helicone.ai`, `llmgateway.io`, `cacheregister.dev`, `blog.pebblous.ai`.

## 5. Known distortions

1. **Token share is dominated by cheap open-weight models.** On Vercel, open-weight models went from 7% of tokens in Dec 2025 to 13% in April, 29% in June, 56% in August and 71.5% in September. The daily record was 78.4% on Sep 18/19. Their spend share stayed at 4–23%.
   - Anthropic's token share therefore fell about 3x between July and September without any fall in Anthropic usage.
   - The same happens at Requesty (open-weight 54.6% of tokens, 14.9% of spend) and OpenRouter.
   - Use the spend share. From September, also ignore request counts: "decision models" took 26% of Vercel requests in October for 0.5% of spend.
2. **Spend is valued at list prices.** Vercel ignores caching and batch discounts. Price cuts (Sonnet 5 on Jun 30, Opus 5.5 on Sep 22, Haiku 5.5 on Oct 7) mechanically lower Anthropic's spend share. That is directionally right for revenue, but the size differs from the true cache-heavy mix.
3. **Anthropic's April 4, 2026 restriction.** From that date, Claude Pro/Max subscription OAuth no longer works in third-party tools. It followed a server-side block on Jan 9 and a legal demand to OpenCode on Mar 19.
   - Since then, Claude use in OpenCode, Kilo, Cline and Roo is API-billed, often through a gateway. Subscription-funded use moved to Claude Code, which is first-party and invisible to every gateway.
   - So there is a structural break in April for every third-party panel. June–October lies after it.
   - It also explains why OpenCode's own data shows Anthropic at about 0.1%, while Requesty's "OpenCode 79% Claude" covers only OpenCode users who pay API prices through Requesty.
4. **Revisions and redefinitions.** Vercel revised August's Anthropic spend share from 64 to 63.7. The September edition redefined "open-weight" using the current model list. History pulled today from the export carries hindsight risk ("有事后风险"); the monthly index posts are the clean vintages.
5. **Edition labels are offset.** The "October 2026" index covers September, and so on. Edition lag after month end: 9–17 days.
6. **Customer mix.** Vercel: web/app developers and v0, now heavy decision-model use. Portkey: enterprises. Requesty and Kilo: coding agents.
7. **Overlap with OpenRouter.** Kilo, Cline and Roo route much of their traffic through OpenRouter, so they are not independent of `openrouter_anthropic_tokens`.
8. **Ceiling effect at Requesty.** Claude is about 92% of coding-agent spend, flat since late 2025, so the series has almost no variation.

## 6. Recommendation

1. **Add `vercel_anthropic_spend_share` now, on the same spec as the OpenRouter indicator.**
   - Monthly; `role = monitor`; wide prior (beta 1 ± 1); drifting intercept.
   - Bounded share, like `github_repo_share_claude`.
   - `published` = the edition date (use `publish_delay_days ≈ 17`).
   - Seven points (Apr–Oct). Treat Oct ~32 as missing until it is verified on the page.
   - Do not use token share as an input.
2. **When `vercel.com` is allowed:**
   - Pull the export for labs and providers, tokens and spend, from the earliest available date.
   - Aggregate it to weekly, mark it hindsight-risk, and check it against the monthly index values.
   - Then save a daily vintage going forward, as is done for Ramp and OpenRouter.
   - The provider split (Anthropic direct vs Bedrock vs Vertex) also helps task C.
3. **Portkey is the second choice.** It gives absolute Anthropic tokens, requests and cost, by direct API vs Bedrock vs Vertex. First check how much history the daily and weekly pages expose, then start daily captures.
4. **Use Kilo's weekly race only as a robustness check,** because it overlaps with OpenRouter. Drop OpenCode, Helicone, LiteLLM, Martian, Cloudflare and Vantage. Keep Requesty only for cost-per-token and model-mix context.
