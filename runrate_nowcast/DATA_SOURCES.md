# 数据盘点：哪些数据能用来建模（2026-11-11）

先盘点数据，再定模型。这次对账暴露的问题是：先行指标（下载量）在 Q3、9/30、10/31 连续高估收入。原因多半不在算法，而在数据缺了一块：**用量要乘以价格才是收入**，现在的模型里没有价格。

## 0. 把收入拆开，看每一块需要什么数据

```
收入 = 按量收费（API 直连 + 云市场 + Enterprise 的用量部分）+ 订阅（Pro / Max / Team / Enterprise 席位）
按量收费 = Σ_模型 token 量 × 每 token 实际价格（标价 × 缓存、批量折扣）
订阅     = Σ_档位 订阅人数 × 月费
run-rate = 按量收费过去 28 天 × 13 + 订阅月费 × 12（S-1 定义）
```

| 需要的量 | 现在有什么 | 质量 | 频率 / 历史 | 能否直接抓 |
|---|---|---|---|---|
| 各模型标价 | **全部模型的价格和发布日期**（[`data/model_prices.csv`](data/model_prices.csv)） | 高，已与官方定价页核对 | 每次发布 / 2025-05 起 | 能 |
| 模型组合（用的是哪个模型） | **GitHub commit 里 Claude Code 写的 co-author 行，带模型名** | 中，只有公开仓库，计数是近似值 | 每周 / 2025-02 起 | 能，但限速很严 |
| token 总量 | 没有官方数字。OpenRouter 每天前 10 名模型的 token 数；SemiAnalysis 的预测 | 低，OpenRouter 只占很小一部分 | 每天 / 只有当天 | 能，需要每天抓 |
| 单任务成本 | Claude Code 官方“每开发者每天”：$6 → $13（2026-04-15 改）；Aider 每轮成本；Artificial Analysis 每任务成本 | 中 | 不定期 | 部分能 |
| 订阅档位比例 | **没有官方数字**。只有价格表、零星新闻和一个刷卡面板数据点 | 低 | — | 价格能抓，人数没有 |
| 按量 / 订阅比例 | **S-1 分拆**：FY2024、FY2025、H1 2026、前三季度 | 高到中 | 年 / 半年 / 季度 | 已录入 |
| 云市场占比 | S-1：2023 11% → 2024 32% → 2025 47% → H1 2026 48% | 中（二手转述） | 年 | 已录入 |
| 大客户数 | S-1：年付 ≥$10 万的客户 20（2023 年底）→ 950（2025-09）→ 11,900（2026-09） | 高（数字在 S-1 原文里） | 年 | 已录入 |
| 消费端需求 | Similarweb 的 claude.ai 访问量（新闻转述）；Ramp；npm / PyPI | 中 | 月 / 周 / 日 | Similarweb 抓不到 |

---

## 1. 各模型的定价（你问的“Claude Code 不同模型后面的定价”）

来源：OpenRouter 的模型 API（能直接访问），其中 14 个模型的价格已逐一与 [官方定价页](https://platform.claude.com/docs/en/about-claude/pricing) 核对，全部一致。最新 4 个模型在官方页面上是前端渲染的，没解析出来，与多个第三方价格汇总一致。

| 发布 | 模型 | 输入 / 输出（$/百万 token） | 相对上一代同档 |
|---|---|---|---|
| 2025-05-22 | Opus 4 / Sonnet 4 | 15 / 75；3 / 15 | |
| 2025-08-05 | Opus 4.1 | 15 / 75 | |
| 2025-09-29 | Sonnet 4.5 | 3 / 15 | |
| 2025-10-15 | Haiku 4.5 | 1 / 5 | |
| 2025-11-24 | Opus 4.5 | 5 / 25 | **Opus −67%** |
| 2026-02 至 05 | Opus 4.6 / 4.7 / 4.8、Sonnet 4.6 | 5 / 25；3 / 15 | 不变 |
| 2026-06-09 | Fable 5 | 10 / 50 | **新的更高一档** |
| 2026-06-30 | Sonnet 5 | 2 / 10 | **Sonnet −33%** |
| 2026-07-24 | Opus 5 | 5 / 25 | 不变 |
| 2026-09-01 | Fable 5.1 | 10 / 50 | 缓存读取降到 0.25 |
| 2026-09-22 | Opus 5.5 | 4 / 20 | **Opus −20%** |
| 2026-09-28 | Sonnet 5.5 | 2 / 10 | 不变 |
| 2026-10-07 | Haiku 5.5 | 0.10 / 0.50 | **Haiku −90%** |
| 2026-11-09 | Fable 5.2 | 8 / 40 | **Fable −20%** |

另外：批量调用一律半价；缓存读取是输入价的 5%–10%；Claude Code 大量使用缓存，所以实际单价明显低于标价。

**与这次高估的关系（假说，待验证）**：Sonnet 在 6/30 降价 33%，正好是 Q3 开始。如果开发者大量换用更便宜的模型，用量（下载、调用次数）就会比收入涨得快，这和 Q3 起指标持续高估的方向一致。下一节的模型组合数据可以检验这个假说。

## 2. 模型组合：GitHub commit 里的模型名

Claude Code 默认在 commit 里写 `Co-Authored-By: Claude <模型名> <noreply@anthropic.com>`，GitHub 的 commit 搜索可以按周、按模型计数：

| 2026-11-01 至 11-07 这一周 | 公开 commit 数 |
|---|---|
| 所有 “Co-Authored-By: Claude” | 约 1,139 万 |
| 其中 Opus 5.5 | 约 678 万 |
| 其中 Sonnet 5.5 | 约 169 万 |
| 对照：“Co-Authored-By: Codex” | 约 8 万 |

最早一周（2025-02-24）只有 3,177 条。

- **能做什么**：算出每周各模型的占比，乘上第 1 节的价格，得到“Claude Code 实际用的模型的加权单价”。这是把用量换成收入时缺的那个价格。
- **不能做什么**：水平值不能直接用。一年多涨了几千倍，远快于收入，因为采用面、提交习惯都在变。只能用占比和变化率，并且要像其他指标一样允许截距漂移。
- **质量问题**：只覆盖公开仓库；用户可以关掉这一行；搜索接口常返回 `incomplete_results = true`，计数是近似值；模型名是后来才写进去的，从哪个版本开始，回填结果会给出。
- **抓取**：2026-11-13 起 GitHub 搜索已认证，但 commit 搜索很耗资源，二级限速很严，实际只能每分钟约 1 次。脚本 [`scripts/github_claude_models.py`](scripts/github_claude_models.py) 慢速抓取、触发限速就退避、可断点续跑。
- **已回填**：
  - 每 4 周的总数：[`data/github_claude_commits.csv`](data/github_claude_commits.csv)（2026-11-11）。
  - 按家族（Opus / Sonnet）的计数：[`data/github_claude_family.csv`](data/github_claude_family.csv)（2026-11-11）。
  - **按模型的计数（2026-11-13）**：[`data/github_claude_models.csv`](data/github_claude_models.csv)，每行带抓取时间。价格研究只用 `price_study.DATA_CUTOFF`（11:40 UTC）之前抓到的行；之后脚本继续补充按模型的细节和模型名出现的时间，这些不进研究。
- **两个计数陷阱（2026-11-13 发现）**：
  1. **GitHub 的短语搜索是前缀匹配。** `"Claude Opus 5"` 也会匹配 `Claude Opus 5.5 ...`。用“父减子”来算，噪声会被放大到不能用：2026-11-02 那周，“Opus 5”减“Opus 5.5”得到 279 万，精确计数只有 15 万。解决办法：名字是别的模型前缀时，在后面接上署名里的下一个词（邮箱），查 `"Co-Authored-By: Claude Opus 5 noreply"`。它只匹配 `Claude Opus 5 <noreply@anthropic.com>`，不匹配 5.5。
  2. **计数本身噪声很大。** 返回值大多带 `incomplete_results = true`。同一个查询隔天再查会差 10%–50%（例如 2026-10-05 那周，Sonnet 5.5 单个模型 142 万，11-11 抓的 Sonnet 家族总数只有 93 万），各家族之和有时比总数还多 50%（2026-09-07 那周）。所以份额有大约 ±15%–20% 的误差。
- **算法**（`pricing.model_counts`）：家族份额来自家族计数；家族内部按精确计数的比例拆分；同一窗口里有两个以上同日抓取的精确计数时，用它们的和作为家族总数，避免混用不同日子的计数。没有拆分数据的家族，全部算给最新发布的那个模型。实测支持这条规则：Opus 4.5 发布一周后（12-01 那周），Opus 4.1 只有 0 条；Opus 4.6 发布三周后（02-23 那周），Opus 4.5 只剩约 4%。只抓了**两个价格不同的相邻版本**（例如 Opus 4.5 与 4.1、Sonnet 5 与 4.6、Opus 5.5 与 5、Haiku 5.5 与 4.5、Fable 5.1 与 5），价格相同的版本没有逐个拆分。
- **实测组合**（每 4 周抽一周；`reports/github_model_counts_2026-11-10.csv`）：

| 抽样周 | Opus | Sonnet | Haiku | Fable | 单价指数：标价 / 缓存口径（12 月 = 1） |
|---|---|---|---|---|---|
| 2025-12-01 | 68%（4.5） | 29%（4.5） | 2% | — | 1.00 / 1.00 |
| 2026-01-26 | 86% | 13% | 2% | — | 1.08 / 1.08 |
| 2026-04-20 | 64%（4.7） | 35%（4.6） | 1% | — | 0.98 / 0.98 |
| 2026-06-15 | 73%（4.8） | 25% | 1% | 2%（5） | 1.06 / 1.06 |
| 2026-07-13 | 40% | 19%（Sonnet 5 占 14 个点） | 1% | **40%** | 1.49 / 1.49 |
| 2026-08-10 | 53%（5） | 23% | 1% | 24% | 1.27 / 1.27 |
| 2026-09-07 | 49% | 31% | 1% | 20% | 1.18 / 1.09 |
| 2026-10-05 | 70%（5.5 占 69） | 22% | 1% | 6% | 0.91 / 0.78 |
| 2026-11-02 | 75%（5.5 占 74） | 20% | 1% | 4% | 0.87 / 0.75 |

  指数是该周样本公开当天（周日之后一天）的单价。缓存口径：每个提示 token 按 0.85 缓存读取 + 0.15 未缓存输入 + 0.015 输出计价（比例取自 OpenRouter 上 Claude 编码流量 2026 年 10–11 月的实测），5.5 系列的缓存读取降价更多，所以 9 月以后两条线分开。7–9 月 Fable（$50/百万输出 token，是 Opus 的两倍）一度占了 20%–40%，所以**按实测组合算的单价在 Q3 是上升的**，10 月 Opus 5.5 降价、Fable 份额回落后才大幅下降。
- **局限**：
  - commit 只反映 Claude Code 用户，其中很多是包月订阅，他们选哪个模型并不改变收入。OpenRouter（按 token 计费）每周前 9 名里从来没有出现过 Fable，这和 commit 里 40% 的 Fable 对不上。也就是说，这个组合更像“订阅用户的组合”，不一定是“按量计费收入的组合”。
  - 计数是事后统一抓的。早期窗口里包含后来才推送的 commit，有修订风险。
  - 每 4 周只抽一周，两次抽样之间的组合变化看不到。
- **模型名什么时候进入署名**（脚本 plan `first`，在研究截止时间之后抓取，只用于说明）：

| 周 | 所有 “Co-Authored-By: Claude” | 带 Opus 名字 | 带 Sonnet 名字 | 带名字的占比 |
|---|---|---|---|---|
| 2025-04-21 至 2025-11-24 的 12 个抽样周 | 0.4 万 – 27.6 万 | 250 – 680 | 12 – 120 | 0.3%–6%（早期总数很小） |
| 2025-12-01 | 29.2 万 | 2.2 万（全是 4.5，Opus 4.1 为 0） | 0.9 万 | 11% |
| 2025-12-08 | 30.2 万 | 15.2 万 | 6.1 万 | 71% |
| 2025-12-15 | 33.7 万 | 20.4 万 | 6.5 万 | 80% |
| 2025-12-22 | 36.3 万 | 22.7 万 | 7.9 万 | 84% |
| 2025-12-29 | 40.0 万 | 31.7 万 | 11.3 万 | 约 100% |

  到 11-24 那周为止，带名字的只有每周几百条，是零星手写的；**Claude Code 从 2025-12-01 那周开始写模型名**（Opus 4.5 发布约一周后），12-08 那周已达 71%，12 月底接近全部。所以实测组合最早从 2025-12-01 那周开始（公开于 12-08），之前用“平价不变”的回退规则（`pricing.FALLBACK`）：2025 年 1 月到 11 月 23 日，各家族标价都没变，单价只可能因组合变化而变，而组合在那之前测不到；11-24 Opus 4.5 降价 67% 的同时用量从 Sonnet 转向 Opus，净效应连符号都不确定（降价前 Opus 占 10%–20% 时，介于约 −12% 和 +13% 之间）。平价不变不添加信息，水平差异由指标的随机游走截距吸收。
- **用作指标的结果**：见 README 2026-11-13 一节（变体 D）。

## 2b. OpenRouter 和 OpenCode（2026-11-11 加入）

**OpenRouter**：公开的前端接口（`/api/frontend/v1/rankings/...`），这个环境能直接访问。快照存在 [`data/openrouter/`](data/openrouter/)。

| 接口 | 内容 | 历史长度 | 用法 |
|---|---|---|---|
| `author-market-share` | 按公司（Anthropic、OpenAI、Google……）统计的每周 token 数 | 滚动 53 周（2025-11 起） | **已进指标目录**：`openrouter_anthropic_tokens`，每周抓取 |
| `model-rankings-chart` | 每周排名前 9 的模型加“其他”的 token 数 | 滚动 53 周 | Anthropic 模型只在进入前 9 时才单列 |
| `task-spend` | 按任务类型（代码、数据、agent……）统计的**花费**份额和 token 份额，最近 30 天，附带与上 30 天的变化 | 两个时点 | Anthropic 占 OpenRouter 花费：最近 30 天 25%，之前 30 天 12.5% |
| `session-cost` | 各编码工具（Claude Code、Codex、Kilo、Hermes）上每个模型的单次会话成本中位数，按会话长短分档 | 最近 30 天 | 单任务成本。Claude Code 的“核心”会话：Opus 4.8 $2.13，Opus 5 $2.12，Opus 5.5 $1.32 |
| `apps` | 应用排行（日、周、月 token） | 当前 | Claude Code 在 OpenRouter 上约 11 万亿 token/周 |

- Anthropic 在 OpenRouter 上的周 token 数：5 月底 5.6 万亿，6 月底 6.5 万亿（峰值），7 月 5.3 万亿，8 月 4.9 万亿，9 月中 4.6 万亿，10 月中 13.4 万亿。**Q3 下滑、10 月暴涨，与收入的方向一致**；SDK 下载量在 Q3 一直在涨。
- 局限：
  - 只有 token，没有花费（花费只在 30 天快照里有）；
  - 历史只有 53 周，必须每周抓取才能积累；
  - OpenRouter 只占 Anthropic 流量的一小部分，而且偏向第三方 agent；
  - 波动幅度比收入大得多。

**OpenCode**（开源、支持多家模型的编码 agent）：

- npm `opencode-ai` 月下载量：2026 年 6 月 926 万，7 月 929 万，8 月 1,082 万，9 月 1,021 万，10 月 1,479 万。Q3 持平、10 月 +45%。**它反映所有厂商的编码 agent 需求，不只 Anthropic**。已进指标目录：`npm_opencode`（monitor）。
- OpenCode 在 OpenRouter 上只有一个第三方网关的小条目（约第 200 名），没有参考价值。
- 它的 GitHub 仓库（不在这个会话的访问范围内）、opencode.ai、models.dev（OpenCode 团队维护的模型价格库，git 历史里有价格变化）都访问不了。

## 2c. 更多另类数据（2026-11-11 晚）

**这个环境能抓、已经存下来的：**

| 数据 | 内容 | Q3 2026 有没有看到放缓 | 用法 |
|---|---|---|---|
| Ramp AI Index 页面里嵌的完整历史 | 每月付费使用各厂商的美国企业比例，2023-07 到 2026-09；另有按行业、公司规模、州的拆分，每人每月 AI 支出，AI 占支付额的比例 | **看到了**。Anthropic 2–4 月每月 +4 到 +7 个百分点，7–9 月每月只 +0.3 到 +1 | 指标 `ramp_adoption_anthropic`（每月，按月末后 20 天公开处理） |
| Ramp 最新一周的 token 花费 | Anthropic 占企业 token 花费 60.5%（截至 10/23 的一周）；与上月同期比，花费 +8.9%，用量 +37.1% | 直接说明了每 token 单价一个月降了约 20% | 只有快照，需要每周保存 |
| GitHub 每月新建的仓库数（提到 claude / openai / gemini） | 2025-01 到 2026-10 | **看到了**。Claude 的份额 4 月 72%，8 月 61%，10 月 70% | 指标 `github_repo_share_claude`（份额，不用总数） |
| npm：`@ai-sdk/anthropic`、`@ai-sdk/openai`、`@ai-sdk/google`、`@langchain/*`、`@anthropic-ai/bedrock-sdk`、`@anthropic-ai/vertex-sdk`、`@google/genai` | 每日下载量，2025 年起 | 没看到。Q3 一路上涨，Vercel AI SDK 里 Anthropic 的份额 6 月 36%，9 月 37% | 已存入 `indicators.csv`，未用作指标 |
| PyPI：`langchain-anthropic`、`langchain-openai` | 最近 180 天 | 没看到（9 月的下降是 PyPI 统计口径的断点） | 已存入 |
| SEC 全文检索：每月提到 Anthropic / OpenAI / Claude Code 的文件数 | 2024-01 到 2026-10 | 每月只有十几到几十份，随财报季大幅波动，用不了 | `data/sec_filing_mentions.csv` |

**这个环境访问不了的**（网络策略拦截，放行申请被拒）：
- Wikipedia 浏览量（wikimedia.org）
- Hacker News（hn.algolia.com）
- Reddit
- Stack Overflow
- Tranco、Cloudflare Radar（网站流行度排名）
- Polymarket、Kalshi、Manifold（预测市场：第三方对收入的实时预测）
- GDELT（新闻量）
- Greenhouse（招聘）
- Vercel AI Gateway、Kilo Code、Cline、Helicone 的用量排行（和 OpenRouter 一样衡量实际用量）
- Similarweb、Appfigures、Sensor Tower、Google Trends
- web.archive.org（历史快照，可以补 OpenRouter 53 周以前和 Ramp 的历史版本）
- artificialanalysis.ai
- Homebrew、VS Code 插件市场
- Hugging Face

**只能从新闻里拿到的零散数字**（二手转述，见 [`data/disclosures.csv`](data/disclosures.csv) 和 README）：
- **Similarweb：** claude.ai 月访问 1 月 2.03 亿，4 月 8.24 亿，5 月 9.53 亿，8 月 9.50 亿。5 月以后基本持平。
- **Appfigures：** Claude 应用 5 月的应用内收入约 $9,200 万；下载量 4 月见顶（约 2,000 万），5 月 1,800 万。AppMagic：7 月下载 1,660 万，比 6 月少 3.5%，收入多 8.8%。
- **Menlo Ventures 2026 报告（11/3）：** Anthropic 占企业 LLM API 花费的 55%，上一年是 40%。
- **Ramp 首席经济学家：** Anthropic 的美元增长在 8–9 月停滞，主要原因是顶级模型降价，不是需求下降。

## 3. 单任务成本随时间的变化

| 数据 | 内容 | 时间范围 | 状态 |
|---|---|---|---|
| Claude Code 官方成本说明 | 企业平均每开发者每活跃日 $6（90% 低于 $12）；2026-04-15 改为 $13（90% 低于 $30），每月 $150–250 | 两个时点 | 当前值已直接读到。历史版本要靠 web.archive.org，这个环境访问不了 |
| Aider polyglot 排行榜 | 每个模型跑完 225 道题的总成本和通过率 | 2024-12 至 2025-10（之后不再更新） | 能抓 |
| Artificial Analysis | 每个模型跑完评测集的成本、每任务成本、token 用量 | 持续更新 | 环境访问不了。新闻摘录：Opus 4.5 约 $1.5k，Opus 4.8 $3,753，Opus 5 $3,836，Fable 5 约 $10k，Fable 5.1 约 $3.7/任务 |

看得出的方向：单 token 价格在降，每个任务用的 token 在涨（Opus 5 单价没变，跑一遍评测反而比 Opus 4.8 略贵）。两者净效应是多少，现有数据还定不了。这正是“用量 × 单价”建模需要的两个分量。

## 4. 订阅档位比例（Pro / Max / Team / Enterprise）

**官方没有拆分。** S-1 只给订阅收入的总数。“paid subscribers”这个词只出现在第一版 DRS 草稿里，后来的版本都删掉了。

| 有的 | 内容 |
|---|---|
| 价格（claude.com/pricing，能抓） | Pro $20（年付 $17）；Max $100（5 倍）/ $200（20 倍）；Team 标准席位 $20–25；**Enterprise $20/席位 + 按 API 价格计用量**，所以 Enterprise 的用量计入按量收费，不计入订阅 |
| 额度变化（影响升级） | 2025-08 加入周额度；2026-03 高峰时段多扣额度；2026-05-06 Claude Code 的 5 小时额度翻倍；临时加 50% 的周额度在 2026-09-14 结束；10/30 Max 改名、取消部分 5 小时限制（未证实） |
| 人数和档位 | 2026 年 3 月新闻：付费订阅今年已翻倍以上，新增大多是 Pro；YipitData 刷卡面板：21.4% 的 Cursor 付费用户同时付费 Claude Max |

**真正能给出档位比例的是刷卡面板数据**（YipitData、Bloomberg Second Measure、Earnest）：扣款金额 $20 / $100 / $200 直接对应档位，按月就有。这类数据要付费。没有它，档位比例只能当作隐变量，靠订阅收入总数和价格去约束，精度有限。

## 5. 按量 / 订阅的比例随时间变化（S-1）

| 期间 | 按量收费 | 订阅 | 订阅占比 | 核实情况 |
|---|---|---|---|---|
| FY2024 | | | 25% | 二手转述 |
| FY2025 | 3.80 | 0.79 | 17% | 二手转述（合计 4.585 与 GAAP 一致） |
| H1 2026 | 13.064 | 3.222 | 19.8% | **两个数字都在 S-1 原文中检索到** |
| 前三季度 2026 | 26.3 | 6.9 | 21% | 二手转述 |
| Q3 2026（推算） | 约 13.2 | 约 3.7 | 约 21.7% | 由前两行相减 |

订阅占比从 2025 年的 17% 回升到 Q3 的约 22%，可能是 Claude Code 带动的订阅（Max）在 2026 年下半年增长更快。现在模型的分部层只用了 4 月的一个份额点，这几个季度数可以直接作为分部层的收入锚点。

全部数字和出处见 [`data/disclosures.csv`](data/disclosures.csv)，每行都标了核实程度。

## 6. 其他需求侧信号

- **Similarweb（claude.ai 访问量，新闻转述）**：2025-08 1.48 亿 → 2026-01 2.03 亿 → 04 8.24 亿 → 05 9.53 亿 → 08 9.50 亿（5 月后持平）；10/22 单日 4,700 万的纪录，28 天合计 11 亿。网页端 5 月后不涨，收入却继续涨，增长多半来自 API 和 Claude Code，而不是聊天网页。
- **OpenRouter**：每天前 10 名模型的 token 数。11/10 当天前 10 名合计 161 万亿，其中 Anthropic 模型 11.9 万亿（7.4%）。月度回顾和周报里有 Anthropic 与 OpenAI 的 token 和花费对比（文字中的数字，没有可下载的序列）。
- **已经在用的**：npm / PyPI 下载量、Ramp AI Index。新发现 PyPI 上的 `claude-agent-sdk` 也能抓。
- **成本侧**：SemiAnalysis 估计订阅只占约 10% 的收入，却用了超过 40% 的推理算力。这影响利润，不影响收入。

## 7. 这个环境能访问什么

| 能访问 | 访问不了（需要在环境设置里放行） | 要付费 |
|---|---|---|
| OpenRouter（模型 API、排行、数据回顾）、GitHub API、raw.githubusercontent.com、claude.com / platform.claude.com / code.claude.com、anthropic.com、npm、PyPI、Ramp、Docker Hub、SEC 全文检索 | web.archive.org（价格页、成本说明、Ramp 的历史版本）、artificialanalysis.ai、formulae.brew.sh（Homebrew 安装数）、marketplace.visualstudio.com / open-vsx.org / plugins.jetbrains.com（IDE 插件安装数）、huggingface.co（Anthropic Economic Index 数据集）、similarweb.com、appfigures.com、epoch.ai、swebench.com、status.anthropic.com、fourweekmba.com | 刷卡面板（YipitData、Second Measure、Earnest）、Similarweb / Sensor Tower 的完整序列 |

SEC 的完整原文需要在 User-Agent 里带联系邮箱；全文检索不需要。

## 8. 数据定了以后怎么建模（建议）

把现在的“下载量 → 收入”换成**按收入恒等式分块建模**：

1. **按量收费 = 用量 × 单价**
   - 单价：第 1 节的标价，按第 2 节的模型占比加权，再乘一个缓存/批量折扣因子（未知，作为缓慢漂移的状态）。
   - 用量：下载量、commit 数、OpenRouter token、Ramp 支出，作为用量的先行指标，各自保留会漂移的截距。
   - 这样，降价会直接压低“同样用量对应的收入”，而不是被当成指标噪声。这一点针对的正是这次连续高估。
2. **订阅 = 订阅收入总量**，用 S-1 的季度订阅收入做锚点。档位比例在有刷卡数据之前只作为隐变量，不单独估计。
3. **两块分别对应 run-rate 定义里的两项**：按量收费按 28 天 × 13，订阅按月费 × 12。官方 run-rate 用这两块加总来对账。

## 9. 优先级（按对精度的帮助和成本排序）

1. **价格表**（已完成）+ **GitHub 模型组合回填**（正在跑）→ 加权单价指数。成本低，直接针对这次的高估。
2. **S-1 分拆**（已录入）→ 给分部层加季度锚点。
3. **放行几个网站**：web.archive.org（历史价格页和成本说明）、artificialanalysis.ai（每任务成本）、formulae.brew.sh 和 IDE 插件市场（Claude Code 安装数）、huggingface.co（Economic Index）。
4. **一个 GitHub token**：回填速度快约 30 倍，可以做每周、按模型的完整序列。
5. **刷卡面板数据**（付费）：订阅档位比例的唯一可靠来源。
6. **SEC 联系邮箱**：读 S-1 原文里的季度分部表和风险因素（比如 “lower prices”）。
