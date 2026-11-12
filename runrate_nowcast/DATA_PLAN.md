# 数据扩充计划（2026-11-12）

目标：找到离收入更近、而且能还原出“当时公开了什么”的数据。规则不变：

- S-1 数字只用来打分，不作输入。
- 每条数据都要有发布日期（`published`），回测时只能用发布日期在时点之前的数据。
- 拿不到当时的历史、只能今天一次性抓的，标为“有事后风险”。
- 不编造数据，抓不到就写抓不到。
- 对外请求不带用户的任何个人信息（User-Agent 用 `runrate-nowcast research script`）。

## 候选数据与状态

| 优先级 | 数据 | 用途 | 当时的历史 | 能否访问 | 负责 | 状态 |
|---|---|---|---|---|---|---|
| 1 | 预测市场：Manifold、Polymarket、Kalshi 上的收入、run-rate、估值盘口 | 基准：每个时点市场预期的中位数和区间；也可作第三方预测输入 | 有：Manifold 有逐笔下注，另两家有价格历史接口 | 接口被网络策略拦截 | 子任务 A | 完成：48 个盘口的清单和抓取代码（`prediction_markets.py`）；价格历史需要放行接口后才能抓 |
| 2 | 第二个用量面板：Vercel AI Gateway、Requesty、Helicone、Portkey 的公开用量排行 | 和 OpenRouter 一起用，抵消单一面板的波动 | 只有滚动窗口快照，需要网页存档补 | 网站被拦截；搜索结果里有带时间窗口的快照 | 子任务 D | 完成：64 条快照（`data/gateway_usage_snapshots.csv`）。Vercel 月报里 Anthropic 的花费份额 4–8 月 61%–65%，9 月 44%，10 月约 32%（未核实） |
| 3 | 云厂商财报：AWS 的 Bedrock、Google Cloud 的 Vertex、Microsoft 的 Foundry 上关于 Claude 和 Anthropic 的说法；Amazon、Google 对 Anthropic 投资的记账 | 云渠道约占收入一半，季度锚点 | 有：按季度，发布日期明确 | SEC 全文检索可用；完整原文需要 SEC 联系邮箱；新闻可搜 | 子任务 B | 完成：144 条（`data/cloud_partner_disclosures.csv`）。只能看出方向，看不出水平 |
| 4 | OpenRouter 上 Claude 各接口的可用率、延迟、吞吐量，以及各供应商（Anthropic 直连、Bedrock、Vertex）的流量占比 | 容量是否卡住收入；云渠道占比 | 接口给当前值；历史需要持续抓 | 可以访问 | 子任务 C | 完成：`openrouter_endpoints.py` 及快照。各供应商占比只有 90 天历史，花费份额只有当前窗口，需要每天抓 |
| 5 | 消费者刷卡面板：YipitData、Second Measure、Earnest、Consumer Edge | 按 $20 / $100 / $200 直接拆订阅档位 | 有 | 付费 | — | 需要预算 |
| 6 | 二级市场和上市后数据：Forge、Caplight 报价；上市后的股价、期权、卖方预测 | 市场预期 | 二级市场只有零星新闻；卖方预测要等 IPO 静默期结束（约 12 月中） | 新闻可搜 | — | 待定 |
| 7 | 关注度：Similarweb、App 数据、Google Trends、Wikipedia、Hacker News、Reddit、招聘岗位 | 消费端热度 | 多数有 | 付费或被拦截 | — | 待放行或预算 |

## 需要放行的网站

在 session 标题栏的环境菜单 → Edit → Allowed domains 里添加：

- api.manifold.markets
- gamma-api.polymarket.com
- clob.polymarket.com
- api.elections.kalshi.com
- vercel.com
- requesty.ai
- web.archive.org

## 验收标准

1. **原始数据**：存在 `data/` 下，每行带来源 URL、发布日期、核实程度。
2. **抓取代码**：可重复运行。接口被拦截的，先写好代码和离线测试，等放行后直接运行。
3. **评估方式事先定好，避免看完结果再挑：**
   - 预测市场先当基准，在每个回测时点比较市场中位数和模型的误差；
   - 新的用量面板按 OpenRouter 的同一规格加入；
   - 不再在已有的 15 个目标上挑最好的版本，而是登记预测，用 Q4 和之后的数字检验。

## 子任务说明

- **A 预测市场**：
  - 写三家接口的抓取代码，并能还原任一时点的概率；
  - 用网页搜索整理相关盘口清单（编号、问题、阈值、结算口径、开盘和结算日期）；
  - 能抓就抓，抓不到如实记录。
- **B 云厂商**：整理 2024Q1–2026Q3 每个季度 Amazon、Alphabet、Microsoft 关于 Bedrock、Vertex、Foundry、Anthropic、Claude 的说法和数字，附发布日期和出处。
- **C OpenRouter 容量和渠道**：
  - 摸清 OpenRouter 公开接口里 Claude 模型的可用率、延迟、吞吐量和各供应商流量占比，写抓取代码并保存快照；
  - 尽量找到可用的历史数据。
- **D 第二个用量面板**：
  - 用网页搜索收集 Vercel AI Gateway、Requesty 等平台带明确时间窗口的公开用量快照（Anthropic 的 token 或花费份额），整理成表；
  - 记录可直接访问的数据地址，等放行后抓取。

## 下一阶段（2026-11-12 起）：模型训练和测试，由子任务执行，我负责验收

| 子任务 | 内容 | 状态 |
|---|---|---|
| E | `features.py`：按时点取特征（Anthropic 和 OpenAI），附防泄露测试 | 进行中 |
| F | `data/openai_claims.csv`：OpenAI 2024–2026 的 run-rate 和收入说法 | 进行中 |
| G | 因子模型加带收缩的回归，用两家公司的面板，按固定流程回测 | 等 E、F 完成 |
| H | 冻结规格，登记 Q4 2026 收入和 12/31 run-rate 的预测 | 等 G 完成 |
