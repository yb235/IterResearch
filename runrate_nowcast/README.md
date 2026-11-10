# runrate_nowcast：用状态空间模型估计 Anthropic 的 run-rate

把“真实 run-rate”当作看不见的状态变量。每条官方披露、媒体泄露和第三方估算，都当作带噪声、可能有偏差的观测。S-1 披露的季度 GAAP 收入是审计口径的锚点，用它约束整条路径，同时反过来检验每一类信源。

```bash
pip install -r runrate_nowcast/requirements.txt
python -m runrate_nowcast nowcast --asof 2026-11-10 --plot   # 报告和图写入 output/
python -m pytest runrate_nowcast/tests -q                    # 5 个测试，约 7 秒
```

---

## 1. 结论速览（截至 2026-11-10）

区间均为 80%。**官方口径**：按 Anthropic 的 run-rate 定义（trailing 28 天 ×13，相当于滞后 14 天），再加上估计出的口径差 b，也就是公司当天若公布会给出的数字。**GAAP 口径**：与 S-1 季度收入一致的瞬时年化收入。

| 日期 | 官方口径 run-rate | GAAP 口径（瞬时） | 月增长（log） |
|---|---|---|---|
| 2026-06-30 | 56.1 [53.0, 59.3] | 58.9 [56.1, 61.8] | +12.5% ± 4.6% |
| 2026-09-30 | **73.2 [68.7, 77.8]** | 74.4 [68.8, 80.5] | +7.1% ± 7.9% |
| 2026-10-31 | **78.7 [70.0, 88.5]** | 80.3 [67.9, 94.9] | +7.2% ± 11% |
| 2026-11-10 | 80.6 [69.5, 93.5] | 82.1 [66.9, 100.9] | +7.0% ± 12% |
| 2026-12-31（预测） | 90.4 [62.9, 130.0] | 91.8 [59.4, 142.0] | |

| 期间 | 收入（$B） | 状态 |
|---|---|---|
| 2026 年 7 月 / 8 月 / 9 月 | 5.30 / 5.74 / 5.92（三者合计 = Q3 GAAP 16.96） | 估计 |
| 2026 年 10 月 | 6.60 [5.85, 7.38] | 估计 |
| **Q4 2026** | **21.4 [16.7, 26.6]** | 预测，已事前登记 |
| FY2026 | 54.6 [49.7, 59.7] | 预测 |

对你的问题的回答：

1. **“9 月底大约 $71–73B”成立。** 模型给出的 9 月收入是 $5.92B，×12 = $71.0B（按月×12 口径）；官方口径为 $73.2B [68.7, 77.8]。Sacra 的 $70B 和 SemiAnalysis 的 $74B 都在区间内。
2. **WSJ 的 “>$80B（10 月底）” 为真的概率约 33%**（用一个从未见过这条报道的模型计算）。达到 $80B 需要 10 月环比涨 9% 左右，而模型估计的增长是 +7% ± 11%，所以有可能，但并不是基准情形。它是单一匿名信源，在模型里按下界观测、给低权重处理，对结果影响很小。
3. **Bloomberg 的“7 月底 >$65B”与 Q3 GAAP 有些冲突**，为真的概率在 29% 到 53% 之间，取决于口径差 b 随时间漂移得多快（见第 5 节的敏感性分析）。要让 Q3 收入只有 $16.96B，同时 7 月中旬瞬时值就已 ≥ $65B，Q3 内的增长就只能很慢。
4. **新发现：官方 run-rate 相对 GAAP 的偏差在漂移。** 把每个官方数字与“只用 GAAP 季度收入推出的路径”比较：

   | 官方说法 | GAAP 推算值 | 偏差 |
   |---|---|---|
   | $9B（2025 年底） | $9.8B | −8% |
   | $14B（2026 年 2 月） | $15.2B | −8% |
   | $19B（2026 年 3 月） | $18.5B | +3% |
   | $30B（2026 年 4 月） | $27.2B | +10% |
   | $47B（2026 年 5 月） | $40.6B | **+15%** |

   2026 年上半年，官方 run-rate 从略低于 GAAP 变成明显高于 GAAP。可能的原因包括：云渠道收入在 run-rate 里按总额算、在 GAAP 里按净额确认；×13 的年化方式放大了高峰期；或者“突破 $X”这类表述本身就挑在高点发布。这正是要用审计数字校准其他信源的原因。

---

## 2. 数据表

完整表格在 [`data/observations.csv`](data/observations.csv)，每行带来源 URL 和备注。字段说明如下：

| 字段 | 含义 |
|---|---|
| `metric` | `run_rate`（年化，$B）或 `revenue`（期间收入，$B） |
| `kind` | `point` 点值；`lower` 下界（“超过 $X”）；`upper` 上界 |
| `ref_date` / `period_start`,`period_end` | 数字对应的日期 / 收入期间 |
| `definition` | run-rate 口径：`t28x13`（Anthropic 官方，滞后 14 天）、`month_x12`、`instant`、`unknown`（滞后 7 天，再加 10 天日期不确定性） |
| `date_sd_days` | 日期不确定性（天），按当时的增长率换算成观测噪声 |
| `published` | 信息首次公开的日期，回测按它冻结信息集 |
| `tier` | `gaap` > `official` > `leak_tier1` > `leak_single` > `third_party` > `secondary` |
| `origin_id` | 去重键。转引同一原始报道的行共用一个 id，只算一次观测 |
| `include` | 0 表示排除（误引、预测性表述等） |

参与估计的观测（去重后共 22 条）：

| 口径日 | 数值 | 类型 | Tier | 来源 |
|---|---|---|---|---|
| 2025-01-01 | $1B | 点 | official | Series F 公告（2025-09-02） |
| 2025-05-30 | $3B | 点 | leak_tier1 | Reuters 独家 |
| 2025-08-31 | >$5B | 下界 | official | Series F 公告 |
| 2025-10-31 | ~$7B | 点 | official | 公司对 Reuters 表示“本月接近 $7B” |
| 2025-12-31 | $9B | 点 | official | Bloomberg 3/3；公司 4/6 声明确认 |
| 2026-02-12 | $14B | 点 | official | Series G 公告 |
| 2026-03-01 | $19B | 点（突破日） | official | Bloomberg 3/3；CEO 在 Morgan Stanley TMT 会议上确认 |
| 2026-04-03 | $30B | 点（突破日） | official | 与 Google/Broadcom 算力协议同发的声明（4/6） |
| 2026-05-12 ± 8 天 | $47B | 点（突破日） | official | Series H 公告（5/28），原文“本月早些时候突破” |
| 2026-07-31 | >$65B | 下界 | leak_tier1 | Bloomberg 8/17（知情人士，投资者更新） |
| Q2 2026 | $11.5B | 收入 | leak_tier1 | 同一份投资者更新里的初步数字 |
| Q1 2026 | $4.73B | 收入 | leak_tier1 | Fortune（9 月）报道的 S-1 草稿 |
| 2026-10-15 ± 20 天 | $70B | 点 | third_party | Sacra（日期近似） |
| 2026-10-15 ± 20 天 | $74B | 点 | third_party | SemiAnalysis（出处不清，见下） |
| 2026-10-31 | >$80B | 下界 | leak_single | WSJ 11/6，单一匿名信源 |
| Q1'25, Q2'25, Q3'25, FY'25 | 0.376, 0.787, 1.382, 4.59 | 收入 | gaap | S-1（10/26） |
| Q1'26, Q2'26 | 4.729, 11.555 | 收入 | gaap | S-1（Q1 由 9M 33.24 − Q2 − Q3 推出） |
| Q3'26 | 16.96 | 收入 | gaap | S-1/A（11/2） |

排除或去重的行：NYT “$100B”（误引，原文是“有望在年内突破”，属于预测）；Reuters 引述的“年底 $9B 内部目标”（目标，不是观测）；Axios、PYMNTS 对 Bloomberg 的转引；Sacra 页面上的“10 月 $80B”（与 WSJ 同源，去重）；Fortune 的 Q2 数字（与投资者更新是同一个公司数字）。

**数据核实状态，请务必看：**
- 这个环境的网络无法直接访问 EDGAR、Sacra、stockanalysis 等原始页面，表中数字来自检索结果对二手报道的汇总。
- Q1/Q2 2025 的拆分只有一个二手来源（两者之和与 9M2025 − Q3 一致）。
- Q3 2026 有两个数字：$16,956M 与 $16.962B。
- Sacra 同时出现 $70B 和 $80B 两个数字。
- SemiAnalysis 的 $74B 一说是“10 月中估算”，一说是“把新季度数据年化”。若是后者，它就是 Q3 GAAP 的函数，不能算独立证据。
- 请把这些行对照 EDGAR 原文核对一遍。

---

## 3. 模型

### 状态（按日步进）

| 状态 | 含义 |
|---|---|
| `l` | log 瞬时年化收入（GAAP 口径） |
| `g` | 月度 log 增长率。Local linear trend：`l' = l + g/30.44`，可选阻尼：`g' = g + κ(ḡ − g)/30.44` |
| `b` | 报告 run-rate 相对 GAAP 的 log 口径差（随机游走） |
| `A_k` | 期间 k 的累计收入。每个被观测或被查询的季度、年、月各有一个累加器，每天加上 `exp(l)/365`，期初清零 |

季度收入因此是状态的一部分，“季度收入等于月度路径的积分”这条约束是精确成立的，不需要事后对账。

### 观测方程（都是状态的线性函数）

- run-rate 说法：`log(y) = l(t − lag) + b + tier_bias + ε`，`ε ~ N(0, σ_tier² + (g·date_sd)²)`
- 收入说法：`y = A_k(期末) + ε`
- “超过 $X” / “不超过 $X”：截断观测。对观测的线性组合做截断正态的矩匹配（assumed-density filtering）。`tests/` 里用拒绝采样核对过，误差在千分之几以内。

### 非线性

`A_k` 是 `exp(l)` 的累加，所以是非线性的。预测步用了精确矩：`E[e^u] = exp(m + v/2)`，再由 Stein 引理得到 `Cov(x, e^u) = P c E[e^u]`，没有做线性化。有一个问题：当某个季度数字离预测很远时（例如 Q3 预测约 $27B、实际 $16.96B），单次高斯更新会让日度路径和累加器对不上，初版模型就算出了 9 月收入低于 8 月。所以平滑用的是 **iterated posterior linearization smoother**：每一轮围绕上一轮的平滑后验做统计线性化，再跑一遍卡尔曼滤波和 RTS 平滑，通常 5–6 轮收敛。

### 参数（[`params.json`](params.json)，由 `fit --write` 生成）

| 参数 | 值 | 怎么来的 |
|---|---|---|
| σ_g（增长冲击） | 0.017 /√天 | MAP 估计，log sd 0.26，识别得较好 |
| κ（增长回归速度） | 0.12 /月（半衰期约 6 个月） | MAP 估计，log sd 0.62 |
| σ_l（水平冲击） | 0.0038 /√天 | 基本由先验决定，数据识别不了 |
| σ_b（口径差漂移） | 0.0019 /√天 | 基本由先验决定，见第 5 节的敏感性分析 |
| 信源噪声 | official 0.04，leak_tier1 0.08，leak_single / third_party 0.15 | 默认值，见下 |

目标函数是在线滤波器的 prediction-error likelihood（真正的一步向前预测误差），加上弱先验。只有约 22 条观测，所以每个参数的识别程度都列了出来。

**信源打分**（`calibrate`）：先只用 GAAP 收入拟合一条“审计真值”路径，再计算每个信源的历史说法相对这条路径的残差，用带收缩先验的 MLE 估计每个 tier 的偏差和噪声。目前只有 official 有 6 个点值，拟合出的 sd 为 0.040，与默认值一致。其他 tier 只有 1–2 个点，不足 3 个点就不覆盖默认值，以免用一两个数据点给信源定权重。这一步和你 bestanalysts 的信源打分加回测是同一套逻辑：样本攒够之后，同一段代码就能自动给出各信源的噪声。

---

## 4. 怎么保证正确

**回测**（`backtest`）：对每个 GAAP 季度和每个官方 run-rate，只用它公布前一天的信息（同源报道一并剔除），预测它的数值。

| | 80% 覆盖率 | 50% 覆盖率 | z 的标准差 | log MAE | 朴素基准 log MAE |
|---|---|---|---|---|---|
| 全样本超参数（n=13） | 85% | 23% | 1.00 | **0.143** | 0.297 |
| 每个时点重新拟合超参数（`--refit`，严格实时） | 77% | 31% | 1.25 | 0.141 | 0.297 |

朴素基准的做法：run-rate 沿用最近一个点值；季度收入把各 run-rate 点值按 log 线性插值后对时间积分。模型误差约为基准的一半，80% 区间基本校准。严格实时版本的 z 的标准差为 1.25，说明区间略偏窄。n=13 太小，50% 覆盖率的波动很大。另外 z 的均值为 +0.3：实际公布的数字往往高于模型的事前预测，与第 1 节发现的官方口径上漂是同一个方向。

**回测暴露的问题**：Q3 数据公布前（10/30），模型对 10 月 run-rate 的预测是 $116B [68, 200]，加阻尼前是 $144B。原因是 local linear trend 会把 Q1–Q2 约 30%/月的增长一路外推。加入增长回归（κ）后，似然和回测都有改善（log MAE 从 0.180 降到 0.143），所以保留了这一项。它仍然说明：**新季度数字出来之前，模型对增速放缓的反应偏慢。** 先行指标层（第 6 节）就是为了补这一块。

**合成数据测试**（`tests/`）：已知真实路径，增速从 30%/月降到 8%/月，口径差 b = 0.06。模型能复现每个季度的收入；80% 区间在 88%–100% 的天数里覆盖真值，略偏保守；b 的估计在真值附近。

**事前登记**（`register` / `score`）：把预测加上时间戳追加到 [`forecasts/registry.jsonl`](forecasts/registry.jsonl)，并记录 git commit 和数据文件的 sha256，然后立即提交。git 历史就是时间戳的凭据。记录内容是未来 120 天的 run-rate 预测扇面，加上未披露期间的收入预测，所以无论下一个数字哪天公布、落在哪个日期，都能对账。第一条记录已经登记，包括 Q4 2026 收入 $21.4B [16.7, 26.6]。

---

## 5. 敏感性：口径差 b 漂移多快

数据几乎识别不了 σ_b：似然轻微偏向慢漂移，回测轻微偏向快漂移。默认用事先定好的 MAP 值，不按回测结果去调。

| σ_b | b（5 月 → 11 月） | 9/30 官方口径 | 10/31 官方口径 | Q4 收入 | P(WSJ >80) | P(BBG >65) | 回测 MAE |
|---|---|---|---|---|---|---|---|
| 0.0019（MAP，默认） | +0.011 → +0.014 | 73.2 [68.7, 77.8] | 78.7 [70.0, 88.5] | 21.4 [16.7, 26.6] | 33% | 29% | 0.143 |
| 0.005 | +0.042 → +0.053 | 75.4 [69.4, 81.8] | 79.6 [70.6, 89.7] | 20.5 [15.7, 25.8] | 35% | 41% | 0.139 |
| 0.010 | +0.084 → +0.086 | 77.3 [69.1, 86.5] | 80.3 [70.9, 91.0] | 19.9 [14.7, 25.6] | 36% | 53% | 0.132 |

10 月底的估计和 WSJ 判断都很稳健，受影响较大的是 Bloomberg $65B 的可信度和 Q4 收入。

---

## 6. 每天怎么更新

```bash
# 1. 抓取和抽取：把新说法追加到 data/observations.csv（日期、数值、上界/下界/点值、口径、tier、origin_id、URL）。
#    多篇文章转引同一条报道时用同一个 origin_id，必须追到原始出处。预测性表述设 include=0。
# 2. 报警：新说法与“公布前一天的模型”比较，|z|>2 或下界成立概率 <2.3% 时报警
python -m runrate_nowcast alerts --since 2026-11-03
# 3. 更新 nowcast
python -m runrate_nowcast nowcast --plot
# 4. 在下一次披露前（例如 Q4 财报、IPO 定价文件）登记预测并提交
python -m runrate_nowcast register --note "pre-Q4 10-K" && git add runrate_nowcast/forecasts && git commit -m "register forecast"
# 5. 披露之后对账；每月重新估计参数
python -m runrate_nowcast score
python -m runrate_nowcast fit --write && python -m runrate_nowcast calibrate
```

单次滤波加平滑（含 6 轮迭代）不到 1 秒；完整报告约 5 秒，回测约 5 秒，严格版回测约 40 秒。

## 7. 还没做的（按价值排序）

1. **先行指标层**：在模型里已有现成接口，只是还没接数据。
   - 算力上限可以作为 `upper` 观测接入。
   - OpenRouter token 份额、Ramp AI Index、Claude Code 代理指标（GitHub 上 Claude 共同署名的 commit、npm 下载量）、AWS/Google 财报中关于 Bedrock/Vertex 的表述，可以作为带系数的观测接入 `g`。
   - 这是缩短“增速放缓要等到下个季度才能看见”这个滞后的唯一办法。
2. **分部层**：把 `l` 拆成 API 直连、云渠道、Claude Code、个人订阅、企业席位，各自有自己的驱动因子。S-1 若披露云渠道按总额还是净额确认，b 的漂移就能部分解释。
3. **抓取和抽取流水线**：这个仓库本身就是一个深度研究 agent（IterResearch），可以让它负责检索、追溯原始出处、抽取成 CSV 行，这个模块负责估计。抽取的格式就是上面的 CSV 字段。
4. **攒信源历史**：每个 tier 至少要 3 个以上点值，才能让 `calibrate --write` 真正给信源定权重。可以把 OpenAI 等公司的 run-rate 说法与其后来的审计数字也纳入同一套打分（跨公司共享 tier 噪声）。

## 8. 主要来源

- S-1/A Q3 数据：[PPC Land](https://ppc.land/anthropic-swings-to-567m-q3-operating-loss-on-5bn-of-short-term-compute/)、[Stock Analysis](https://stockanalysis.com/news/anthropic-third-quarter-revenue-17-billion-ipo-filing/)、[Let's Data Science](https://letsdatascience.com/news/anthropic-reports-17-billion-q3-revenue-in-amended-s-1-9b752413)
- S-1 上半年和 2025 年数据：[The Next Web](https://thenextweb.com/news/anthropic-ipo-prospectus-42bn-loss-518bn-compute)、[Allocations](https://www.allocations.com/insights/anthropic-quarterly-revenue-what-the-s-1-shows)、[PPC Land](https://ppc.land/anthropic-files-for-nasdaq-ipo-with-16-3bn-in-first-half-revenue/)
- run-rate 定义（trailing 28 天 ×13 加订阅 ×12）：[Simon Willison](https://simonwillison.net/2026/May/31/anthropic-run-rate/)、[Let's Data Science](https://letsdatascience.com/news/anthropics-run-rate-revenue-raises-accounting-comparability-9def5524)
- 官方披露：[Series F](https://www.anthropic.com/news/anthropic-raises-series-f-at-usd183b-post-money-valuation)、[Series G](https://www.anthropic.com/news/anthropic-raises-30-billion-series-g-funding-380-billion-post-money-valuation)、[Series H](https://www.anthropic.com/news/series-h)、[$30B（VentureBeat）](https://venturebeat.com/technology/anthropic-says-it-hit-a-30-billion-revenue-run-rate-after-crazy-80x-growth)
- 媒体：[Bloomberg $19B](https://www.bloomberg.com/news/articles/2026-03-03/anthropic-nears-20-billion-revenue-run-rate-amid-pentagon-feud)、[Bloomberg $65B](https://news.bgov.com/financial-accounting/anthropic-revenue-run-rate-surpasses-65-billion-ahead-of-ipo)、[Axios](https://axios.com/2026/08/17/anthropic-revenue-run-rate-ipo-openai)、[Reuters $3B](https://www.thestar.com.my/tech/tech-news/2025/05/31/exclusive-anthropic-hits-3-billion-in-annualized-revenue-on-business-demand-for-ai)、[Reuters ~$7B](https://www.thestar.com.my/tech/tech-news/2025/10/16/exclusive-anthropic-aims-to-nearly-triple-annualized-revenue-in-2026-sources-say)、WSJ $80B 经由 [Crypto Briefing](https://cryptobriefing.com/anthropic-revenue-run-rate-80-billion-ipo/) 转述
- 第三方：[Sacra](https://sacra.com/c/anthropic/)
