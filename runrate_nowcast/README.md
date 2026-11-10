# runrate_nowcast：用三层状态空间模型估计 Anthropic 的 run-rate

**审计数字定水平，先行指标看变化，信源说法补空档。** 三层放在同一个卡尔曼滤波器里：

| 层 | 状态 | 数据 | 作用 |
|---|---|---|---|
| 总量 | log run-rate `l`、增长 `g`、口径差 `b` | S-1 季度 GAAP 收入（锚点）、官方 run-rate、泄露 | 定水平 |
| 分部 | 消费 / 订阅（分割，份额和为 1）；云渠道、Claude Code（叠加层） | S-1 H1 拆分（消费 80% / 订阅 20%，云渠道 48%）、Claude Code 官方数字 | 拆结构 |
| 先行指标 | 每个指标一个**会漂移的截距** `a_i` | npm / PyPI SDK 下载、Ramp 企业 token 支出（每周） | 看变化 |

先行指标的观测方程是 `log(指标) = a_i + β_i·log(分部收入) + 噪声`，其中 `a_i` 是随机游走。指标与收入的关系不假定固定，而是每天都允许变化，审计数字一到就重新校准。

```bash
pip install -r runrate_nowcast/requirements.txt
python -m runrate_nowcast update-data                        # 抓取 npm / PyPI / Ramp
python -m runrate_nowcast nowcast --asof 2026-11-10 --plot   # 报告和两张图写入 reports/
python -m runrate_nowcast layers                             # 逐层回测（约 25 秒）
python -m pytest runrate_nowcast/tests -q                    # 10 个测试，约 9 秒
```

---

## 0. “用历史数据预测未来”是不是假的？回测给出的回答

| 变体（同一组 13 个目标，只用公布前一天的信息） | 80% 覆盖率 | log 误差 |
|---|---|---|
| A 只用总量信源说法（v1） | 77% | 0.138 |
| B + 分部说法 | 92% | 0.137 |
| **C + 先行指标（默认）** | **85%** | **0.081（−41%）** |
| D + 公司 / 第三方预测 | 77% | 0.086 |
| C 但先行指标不标定（用目录里的先验参数） | 31% | 0.228 |
| 朴素外推（最近一个官方数字） | — | 0.297 |

1. **你说得对的一半：单靠收入趋势外推，在拐点处会失效。** v1 在 Q3 数据出来之前预测 10 月 run-rate 为 $116–144B，因为它把上半年约 30%/月的增速一直外推。先行指标在两次披露之间提供了真实的“当期”信息，回测误差下降 41%。改善主要出现在官方数字稀疏的时段：2025 年 10 月从 0.46 降到 0.15，2026 年 2 月从 0.39 降到 0.09。
2. **需要补充的一半：另类数据同样要用历史来标定。** 不标定的 SDK 下载放进模型比完全不用还差（0.228）。原因是下载量每周的变化约为 ±10%，而且变化本身没有自相关，也就是说它有自己的随机游走，不是“收入加白噪声”。历史数据在这里的用途不是外推收入，而是估计每个指标的弹性 β 和漂移速度。拟合出的结果是：npm SDK 的 β≈1.0，漂移 0.033/√天；PyPI 的 β≈1.0，漂移 0.014/√天。参数取整数附近的值（β=0.85，漂移 0.01–0.03）时误差为 0.073–0.075，所以收益不依赖精确拟合值。
3. **公司和第三方的预测：默认不放进滤波器，单独记分。** 放进去后误差反而略升（0.081 → 0.086）。
   - 公司 2025 年 10 月对“2026 年底 run-rate”的内部目标是 $20–26B，模型现在估计约 $134B，只有实际的 1/5。
   - 同一时间对“2025 年底 $9B”的短期目标则很准（0.96）。
   - FT 8 月的投资者预期 $100–120B，与模型的差距是 0.82。

   预测类说法的价值取决于预测期长短和来源的历史命中率，所以先攒记录（报告里的 “Forecast claims” 表），等证明可靠后再按预测期给权重。
4. **分部层不提高总量精度（A→B 基本不变），但 80% 覆盖率从 77% 提高到 92%，并给出结构信息。** 只有消费 / 订阅这一组拆分经过审计，云渠道和 Claude Code 是根据披露和估算推出来的，区间较宽。

---

## 1. 结论速览（三层模型，截至 2026-11-10）

完整报告：[`reports/nowcast_2026-11-10.md`](reports/nowcast_2026-11-10.md)；图：[`reports/nowcast_2026-11-10.png`](reports/nowcast_2026-11-10.png)、[`reports/layers_2026-11-10.png`](reports/layers_2026-11-10.png)。区间均为 80%，**官方口径** = Anthropic 定义（trailing 28 天 ×13）加口径差 b。

| 日期 | 官方口径 run-rate | 月增长（log） |
|---|---|---|
| 2026-09-30 | 75.1 [70.9, 79.6] | +19% ± 6% |
| 2026-10-31 | **91.5 [84.0, 99.6]** | +20% ± 7% |
| 2026-11-10 | **98.0 [88.8, 108.1]** | +20% ± 9% |
| 2026-12-31（预测） | 134.3 [106.0, 170.1] | |

| 期间 | 收入（$B） | 对照：v1（无先行指标） |
|---|---|---|
| 2026 年 7 / 8 / 9 月 | 5.12 / 5.70 / 6.14（合计 = Q3 GAAP 16.96） | 5.30 / 5.74 / 5.92 |
| 2026 年 10 月 | 7.67 [7.11, 8.25] | 6.60 [5.85, 7.38] |
| **Q4 2026** | **28.2 [24.2, 32.4]** | 21.4 [16.7, 26.6] |
| FY2026 | 61.4 [57.4, 65.6] | 54.6 [49.7, 59.7] |

**两个版本的 Q4 预测都已在 [`forecasts/registry.jsonl`](forecasts/registry.jsonl) 登记**（区间不重叠的部分很大），Q4 数字公布后可以直接判断先行指标是否带来了真实的精度。

| 分部（官方口径，11-10） | run-rate | 份额 |
|---|---|---|
| 消费（API、云渠道、按量计费的应用） | 78.8 [71.3, 87.2] | 80.5% |
| 订阅（Pro/Max/Team/企业席位） | 19.0 [16.6, 21.8] | 19.4% |
| 其中经云市场计费（叠加层） | 47.0 [41.0, 53.8] | 48% |
| Claude Code（叠加层） | 16.5 [11.2, 24.3] | 17% [11%, 25%] |

| 先行指标（最近 4 周） | 指标变化 | 换算成收入 | 模型采信的收入变化 |
|---|---|---|---|
| npm `@anthropic-ai/sdk` | +22.3% | +22.3% | +18.7% |
| PyPI `anthropic`（已处理 8/25 统计口径断点） | +18.9% | +18.9% | +18.7% |
| Ramp 企业 token 支出（9/25→10/23） | +25.0% | +30.0% | +18.4% |

三个独立来源都显示 10 月明显再加速（OpenRouter 的模型列表显示新一代 Claude 模型在 9/21–10/7 上线，时间上吻合）。由此带来的变化：
- **WSJ “10 月底 >$80B” 为真的概率**：从 v1 的 33% 升到 **77%**。
- **Bloomberg “7 月底 >$65B”**：降到 16%。指标显示 7 月水平偏低、9 月偏高，Q3 内部的增长更陡。
- **Claude Code “2 月 >$2.5B”**：概率只有 8%，说明 Claude Code 的份额上升速度超过模型的随机游走假设。这是下一步要给分部加趋势项的信号。

### 数据质量记录

- **PyPI 断点**：2026-08-25 anthropic 和 openai 包的下载量同时下降约 35%，是 pypistats 统计口径的变化。已在目录里登记为 `breaks`，滤波器在当天放大截距方差，把这个台阶吸收掉。测试显示，不处理的话会被误读成收入下降 39%。
- **npm 的 Claude Code 下载**：随版本发布出现尖峰，且部分用户已转到原生安装器，所以只作监测，不进滤波器。
- **Ramp 和 OpenRouter 网页只显示当期加上一期**，没有历史。需要每周抓一次，自己积累历史。
- **SEC EDGAR**：全文检索可以访问，查到了 S-1/A（11/2）和 **今天（11/10）新提交的 S-1/A 第 2 号修订**，看起来离定价很近。但 SEC 要求 User-Agent 里带联系邮箱，原文暂时下载不了，见第 7 节。

## 1b. v1 结论（只用信源说法，不含先行指标；保留作对照）

以下数字来自第一版（commit `d2f72eb`），其 Q4 预测已在 `forecasts/registry.jsonl` 第一条登记。

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

   2026 年上半年，官方 run-rate 从略低于 GAAP 变成明显高于 GAAP。可能的原因包括：run-rate 与 GAAP 对折扣、赠送额度的处理不同（S-1 显示云渠道收入在 GAAP 里也按总额确认，所以不是总额/净额差异）；×13 的年化方式放大了高峰期；或者“突破 $X”这类表述本身就挑在高点发布。这正是要用审计数字校准其他信源的原因。

---

## 2. 数据表

完整表格在 [`data/observations.csv`](data/observations.csv)，每行带来源 URL 和备注。字段说明如下：

| 字段 | 含义 |
|---|---|
| `metric` | `run_rate`（年化，$B）或 `revenue`（期间收入，$B） |
| `kind` | `point` 点值；`lower` 下界（“超过 $X”）；`upper` 上界 |
| `ref_date` / `period_start`,`period_end` | 数字对应的日期 / 收入期间 |
| `definition` | run-rate 口径：`t28x13`（Anthropic 官方，滞后 14 天）、`month_x12`、`instant`、`unknown`（滞后 7 天，再加 10 天日期不确定性） |
| `date_lo` / `date_hi` / `date_rule` | 原文措辞允许的时间窗口和措辞依据（例如“本月早些时候突破” → 5/1–5/28）。有窗口时，日期取中点，`date_sd_days` = 窗口宽度 / √12（窗口内均匀分布），不再手填 |
| `date_sd_days` | 日期不确定性（天），按当时的增长率换算成观测噪声。没有窗口的行（审计数字、指标）为 0 |
| `published` | 信息首次公开的日期，回测按它冻结信息集 |
| `tier` | `gaap` > `official` > `leak_tier1` > `leak_single` > `third_party` > `secondary` |
| `origin_id` | 去重键。转引同一原始报道的行共用一个 id，只算一次观测 |
| `include` | 0 表示排除（误引等） |
| `segment` | `total` 或分部名（`subscription`、`cloud`、`claude_code`…） |
| `is_forecast` | 1 = 预测类说法（公司目标、投资者预期），只记分，不进滤波器 |

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
| `s_k` | 分部 k 的 log 份额（随机游走）。消费和订阅是分割，每周施加一次软约束 Σexp(s_k)=1，非线性部分按统计线性化处理；云渠道和 Claude Code 是叠加层，不受约束 |
| `a_i` | 先行指标 i 的 log 截距（随机游走，漂移速度单独估计）。目录里登记的断点日期给截距加一次大方差，用来吸收统计口径变化 |

季度收入因此是状态的一部分，“季度收入等于月度路径的积分”这条约束是精确成立的，不需要事后对账。

### 观测方程（都是状态的线性函数）

- run-rate 说法：`log(y) = l(t − lag) + b + tier_bias + ε`，`ε ~ N(0, σ_tier² + (g·date_sd)²)`
- 收入说法：`y = A_k(期末) + ε`
- 分部份额（例如 H1 订阅占 19.6%）：`log(y) = s_k + ε`，期间平均份额近似为收入加权中点的份额，额外加 0.03 的噪声
- 分部 run-rate（例如 Claude Code $1B）：`log(y) = l + s_k + b + ε`
- 先行指标（每周均值）：`log(I) = a_i + β_i·(l + s_seg) + ε`
- 预测类说法（`is_forecast=1`）：噪声按预测期加大 0.08/月；默认不进滤波器
- “超过 $X” / “不超过 $X”：截断观测。对观测的线性组合做截断正态的矩匹配（assumed-density filtering）。`tests/` 里用拒绝采样核对过，误差在千分之几以内。

### 非线性

`A_k` 是 `exp(l)` 的累加，所以是非线性的。预测步用了精确矩：`E[e^u] = exp(m + v/2)`，再由 Stein 引理得到 `Cov(x, e^u) = P c E[e^u]`，没有做线性化。有一个问题：当某个季度数字离预测很远时（例如 Q3 预测约 $27B、实际 $16.96B），单次高斯更新会让日度路径和累加器对不上，初版模型就算出了 9 月收入低于 8 月。所以平滑用的是 **iterated posterior linearization smoother**：每一轮围绕上一轮的平滑后验做统计线性化，再跑一遍卡尔曼滤波和 RTS 平滑，通常 5–6 轮收敛。

### 参数（[`params.json`](params.json)，由 `fit --write` 和 `fit-indicators --write` 生成）

先行指标参数（MAP，先验来自 [`data/indicator_catalog.csv`](data/indicator_catalog.csv)）：

| 指标 | β（弹性） | 截距漂移 /√天 | 周观测噪声 |
|---|---|---|---|
| npm `@anthropic-ai/sdk` | 1.00 | 0.033 | 0.031 |
| PyPI `anthropic` | 1.00 | 0.014 | 0.019 |
| Ramp token 支出 | 0.83 | 0.005（先验） | 0.089 |

总量层参数（v1 的数字；加入指标后重估：σ_g 0.017、κ 0.15，变化不大）：

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

> 本节是 v1（只用信源说法）的回测记录。三层模型的逐层回测见第 0 节，命令是 `python -m runrate_nowcast layers`。

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
# 0. 先行指标：每天抓 npm / PyPI，每周抓 Ramp 快照（PyPI 只保留 180 天，Ramp 只显示两期，必须持续抓才有历史）
python -m runrate_nowcast update-data
# 1. 抓取和抽取：把新说法追加到 data/observations.csv（日期、数值、上界/下界/点值、口径、tier、origin_id、URL、segment、is_forecast）。
#    多篇文章转引同一条报道时用同一个 origin_id，必须追到原始出处。预测性表述设 is_forecast=1（单独记分，不进滤波器）。
# 2. 报警：新说法与“公布前一天的模型”比较，|z|>2 或下界成立概率 <2.3% 时报警
python -m runrate_nowcast alerts --since 2026-11-03
# 3. 更新 nowcast
python -m runrate_nowcast nowcast --plot
# 4. 在下一次披露前（例如 Q4 财报、IPO 定价文件）登记预测并提交
python -m runrate_nowcast register --note "pre-Q4 10-K" && git add runrate_nowcast/forecasts && git commit -m "register forecast"
# 5. 披露之后对账；每月重新估计参数
python -m runrate_nowcast score
python -m runrate_nowcast fit --write && python -m runrate_nowcast fit-indicators --write && python -m runrate_nowcast calibrate
# 6. 新加一个指标：在 indicator_catalog.csv 里先设 role=monitor，跑 layers 看回测是否改善，再改成 model
```

含先行指标后，单次滤波加平滑约 2 秒，完整报告约 10 秒，逐层回测约 25 秒，指标参数拟合约 20 秒。

## 7. 下一步（按对精度的价值排序）

1. **接近收入的支出类数据，优先于下载量这类活动数据。** 回测里最有价值的是“离钱近”的数据。Ramp 的 β=0.83 而噪声小，下载量的漂移则大得多。候选来源：
   - Ramp：每周抓取，自己积累历史；可以用 web.archive.org 回填，需要放行该主机。
   - 付费卡支出面板：YipitData、Bloomberg Second Measure、Earnest。
   - 云市场：AWS / Google 财报里关于 Bedrock / Vertex 的表述。
2. **SEC 原文。** EDGAR 要求 User-Agent 带联系邮箱。你给一个可以用的邮箱后，就能下载 S-1/A 原文，核对表里所有二手数字，并检查今天的第 2 号修订里有没有“近期经营情况”（10 月初步数字）。
3. **分部加趋势项。** Claude Code 份额的上升速度超过随机游走的假设（“2 月 >$2.5B”的概率只有 8%），应给叠加层加一个份额趋势状态。
4. **GitHub 上 Claude 共同署名的 commit**（Claude Code 的代理指标）：抓取器已写好，但 API 限流严，计数是近似值，需要每周抽一天、慢速回填。
5. **算力上限**：作为 `upper` 观测接入（已支持上界）。
6. **抓取和抽取流水线**：这个仓库本身就是一个深度研究 agent（IterResearch），可以让它负责检索、追溯原始出处、抽取成 CSV 行；`data/` 里两个 CSV 的字段就是接口。
7. **攒信源和预测的历史。** 每个 tier 至少要 3 个点值才会覆盖默认噪声。预测类说法按预测期记分，等可靠后再给权重。

## 8. 主要来源

- S-1/A Q3 数据：[PPC Land](https://ppc.land/anthropic-swings-to-567m-q3-operating-loss-on-5bn-of-short-term-compute/)、[Stock Analysis](https://stockanalysis.com/news/anthropic-third-quarter-revenue-17-billion-ipo-filing/)、[Let's Data Science](https://letsdatascience.com/news/anthropic-reports-17-billion-q3-revenue-in-amended-s-1-9b752413)
- S-1 上半年和 2025 年数据：[The Next Web](https://thenextweb.com/news/anthropic-ipo-prospectus-42bn-loss-518bn-compute)、[Allocations](https://www.allocations.com/insights/anthropic-quarterly-revenue-what-the-s-1-shows)、[PPC Land](https://ppc.land/anthropic-files-for-nasdaq-ipo-with-16-3bn-in-first-half-revenue/)
- run-rate 定义（trailing 28 天 ×13 加订阅 ×12）：[Simon Willison](https://simonwillison.net/2026/May/31/anthropic-run-rate/)、[Let's Data Science](https://letsdatascience.com/news/anthropics-run-rate-revenue-raises-accounting-comparability-9def5524)
- 官方披露：[Series F](https://www.anthropic.com/news/anthropic-raises-series-f-at-usd183b-post-money-valuation)、[Series G](https://www.anthropic.com/news/anthropic-raises-30-billion-series-g-funding-380-billion-post-money-valuation)、[Series H](https://www.anthropic.com/news/series-h)、[$30B（VentureBeat）](https://venturebeat.com/technology/anthropic-says-it-hit-a-30-billion-revenue-run-rate-after-crazy-80x-growth)
- 媒体：[Bloomberg $19B](https://www.bloomberg.com/news/articles/2026-03-03/anthropic-nears-20-billion-revenue-run-rate-amid-pentagon-feud)、[Bloomberg $65B](https://news.bgov.com/financial-accounting/anthropic-revenue-run-rate-surpasses-65-billion-ahead-of-ipo)、[Axios](https://axios.com/2026/08/17/anthropic-revenue-run-rate-ipo-openai)、[Reuters $3B](https://www.thestar.com.my/tech/tech-news/2025/05/31/exclusive-anthropic-hits-3-billion-in-annualized-revenue-on-business-demand-for-ai)、[Reuters ~$7B](https://www.thestar.com.my/tech/tech-news/2025/10/16/exclusive-anthropic-aims-to-nearly-triple-annualized-revenue-in-2026-sources-say)、WSJ $80B 经由 [Crypto Briefing](https://cryptobriefing.com/anthropic-revenue-run-rate-80-billion-ipo/) 转述
- 第三方：[Sacra](https://sacra.com/c/anthropic/)
- 先行指标：[npm downloads API](https://api.npmjs.org/downloads/range/2025-01-01:2026-11-09/@anthropic-ai/sdk)、[pypistats](https://pypistats.org/api/packages/anthropic/overall)、[Ramp AI Index](https://ramp.com/data/ai-index)、[OpenRouter rankings](https://openrouter.ai/rankings)
- 分部：S-1 收入拆分 [Luminix 摘要](https://www.useluminix.com/reports/company-overviews/anthropic-revenue-and-burn-rate/source/2)、云渠道 48% [FourWeekMBA](https://fourweekmba.com/ai-clouds-bill-48-of-anthropic-revenue-it-keeps-the-customer/)、Claude Code $8B 估算 [FourWeekMBA](https://fourweekmba.com/ai-claude-code-run-rate-agentic-layer/)
- 预测：[Reuters 2025-10-16](https://www.thestar.com.my/tech/tech-news/2025/10/16/exclusive-anthropic-aims-to-nearly-triple-annualized-revenue-in-2026-sources-say)
