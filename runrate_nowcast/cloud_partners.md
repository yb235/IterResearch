# 云厂商披露：能否作为云渠道的季度锚点（子任务 B）

数据：`data/cloud_partner_disclosures.csv`，144 行，2024Q1–2026Q3。其中 Amazon 70 行，Alphabet 40 行，Microsoft 34 行。

- 核实程度：
  - `EDGAR phrase confirmed`（72 行）：原句或数字串经 SEC 全文检索确认出现在该文件里。只确认了数字串的，`notes` 里会说明数字的含义来自搜索摘要。
  - `page read`（40 行）：原文已读。Microsoft 全部 11 次财报会议的逐字稿和新闻稿都在 microsoft.com 上读过；Anthropic 和合作方的联合公告在 anthropic.com 上读过。
  - `search summary`（32 行）：只来自搜索摘要。
- `fiscal_period` 一律用自然季度：
  - Microsoft 的财年到 6 月结束，FY26 Q2 对应 2025Q4，`notes` 里写了原财季。
  - 财报类条目标所报告的季度；投资、算力协议等公告标发布当季。
- `value` 只填明确数字。原文是“多十亿”“三位数增长”“翻倍以上”这类说法的，按下限填（如 100、200），并在 `unit` 里注明是下限。

## 一、三家对 Anthropic 点名的程度

| 公司 | 正式文件里的 Anthropic | 财报会议 |
|---|---|---|
| Amazon | 2024–2026 每份 10-Q、10-K 都写到 Anthropic 投资（可转债、无投票权优先股、公允价值、收益）；财报新闻稿除 2024Q4 外每季都提到 Claude（Bedrock 上新增的模型） | Jassy 多次谈 Bedrock、Trainium、Project Rainier，但从不拆分 Claude 的收入 |
| Alphabet | 2024-01 至 2026-11 的 10-K、10-Q、8-K 里“Anthropic”和“Claude”都是 0 次命中（EDGAR）。投资收益只写成“一家私人公司”“SpaceX and a private company” | 确认点名的只有 2025Q3 一次（Pichai 提到 Anthropic 要用最多 100 万颗 TPU） |
| Microsoft | 只有三份：FY24 10-K（列为竞争对手）、2024-12-03 的一份 8-K 附件（内容未读）、FY26 Q4 新闻稿（Anthropic 投资收益 32 亿美元） | FY26 Q2 起每季提到 Foundry 上的 Anthropic 模型；FY26 Q2 说商业订单 +230% 部分来自 Anthropic 的 Azure 承诺 |

## 二、候选季度锚点

| 序列 | 已有观测 | 和 Anthropic 收入的关系 | 主要问题 |
|---|---|---|---|
| **Bedrock 增长**（AWS） | 2025Q4：多十亿美元年化，客户支出环比 +60%；2026Q1：环比 +170%；2026Q3：财富 500 强的 Bedrock 支出同比约 7 倍，token 约 17 倍。在 Bedrock 上用 Claude 的客户：数万（2024-11）→ 超过 10 万（2026-04） | 最接近“通过 AWS 卖出的 Claude”，也就是 Anthropic 的收入 | 口径每季都变（环比支出 → 财富 500 强同比）；Claude 在 Bedrock 里的占比没披露；只有好看时才披露 |
| **Microsoft Foundry** | FY26 Q2：超过 1,500 家客户同时用过 Anthropic 和 OpenAI 模型；FY26 Q3：这个数环比翻倍；FY26 Q4：Foundry 收入同比翻倍以上，10 万家客户；FY27 Q1：收入同比三倍以上，12 万家以上客户 | Claude 从 2025-11 起才在 Foundry 上架。Microsoft 自己买 Claude（The Information 称约 5 亿美元/年）是 Anthropic 的直接收入 | Foundry 以 Azure OpenAI 为主，没有按模型拆分；Microsoft 从 Anthropic 模型销售中抽成的比例不明 |
| **AI 业务年化收入** | AWS：超过 150 亿美元（2026-04-09 股东信，以及 Q1 财报会）→ 超过 250 亿美元（2026Q2）；Microsoft：超过 370 亿美元（FY26 Q3） | 只能当上限或背景 | 包含 Anthropic 自己在 AWS 上买算力的支出（Project Rainier、Trainium）。这是 Anthropic 的成本，方向相反 |
| **投资账面价值和收益** | Amazon 税前收益：33 亿（2025Q1，转换产生）、95 亿（2025Q3）、168 亿（2026Q1）、534 亿（2026Q2，大部分来自 Anthropic）。可转债公允价值：138 亿（2024 年末）→ 458 亿（2025 年末）→ 422 亿（2026Q1，部分转成优先股）→ 979 亿（2026Q2）→ 1,462 亿（2026Q3）。Microsoft：32 亿（FY26 Q4）。Alphabet：未点名的私人公司收益（以上均为美元） | 只反映估值，不是收入 | 时间点由融资轮次决定（Series F、G、H），而融资轮次本身早已公开，几乎没有新信息。会计处理也有影响：可转债的变动进其他综合收益，优先股只在有可观察价格时才调整。2026Q3 可转债价值 +483 亿美元，但当季没有新一轮融资，可能反映了 IPO 预期（三级估值） |
| **云业务增速和积压订单** | AWS 同比 17% → 39%；Google Cloud 28% → 103%；Azure 29% → 48%。积压订单：AWS 3,640 亿美元（不含 Anthropic 超过 1,000 亿美元的新协议）→ 5,780 亿；Google Cloud 1,060 亿（会议口径）→ 5,608 亿（10-Q 里的公司整体剩余履约义务）；Microsoft 商业订单 +230%（含 Anthropic 的 300 亿美元承诺） | 大部分是 Anthropic 作为买家的算力承诺 | 可以用来看 Anthropic 算力是否卡住收入（容量约束），不能当收入锚点 |

**结论**：可以直接当作“云渠道收入”季度锚点的序列，只有 Bedrock 增长和 Foundry 增长两个。它们都是定性或下限式的说法，口径也不连续，适合用来判断方向（加速还是减速），不适合定水平。其余序列要么是 Anthropic 的成本（算力、积压订单），要么只反映估值（投资收益）。

## 三、时效

- **发布时间**：三家都在季末后 22–37 天发布季报（Alphabet 和 Microsoft 在 4、7、10 月的最后一周，以及 2 月初；Amazon 晚一两天），10-Q 再晚一天。季度内的额外信息来源有：Amazon 股东信（4 月初）、合作公告（如 2025-10-23 TPU、2025-11-18 Azure、2026-04-20 AWS 5GW）、媒体报道。
- **和 Anthropic 自己披露的对比**：Anthropic 的年化收入说法不定期出现在季度中（如 2025-10-23 超过 30 万家企业客户；2026-04-06、04-20 年化收入超过 300 亿美元，2025 年末约 90 亿美元），S-1（2026-10-26）给到 2026 上半年。云厂商的数据晚，但按固定日程发布，适合做季度校验。
- **2026Q3**：三家在 10-28 至 10-30 发布的 Q3 数据，是目前唯一官方覆盖 7–9 月云渠道的数据（S-1 只到上半年；S-1 里有没有 Q3 初步数字，本次没查）。
- **回测**：发布日期明确，原文不会事后改写，事后风险低。但搜索摘要多次把不同季度混在一起（如把 2025Q3 的 95 亿美元收益当成 2026Q3 的；把 2025Q3 的积压订单写成 2026Q3 的）。这些在 `notes` 里标出了，用之前最好再核一次原文。

## 四、缺口和被拦截的来源

- 被拦截：ir.aboutamazon.com、aboutamazon.com、abc.xyz、blog.google、各大转录站（fool.com、marketbeat、seekingalpha 等）和新闻站。www.sec.gov 原文按要求没有下载。能访问的只有 efts.sec.gov（全文检索）、www.microsoft.com、www.anthropic.com、cloud.google.com。所以 Amazon 和 Alphabet 财报会上的原话只有搜索摘要级别。
- 没找到：
  - Amazon 2025Q2、2025Q4 单季的 Anthropic 收益；
  - 2025 年各季末的可转债公允价值；
  - AWS 在 2026Q3 的 AI 年化收入；
  - Alphabet 季报会议上关于 Anthropic 的更多原话；
  - Microsoft 削减 Claude 支出那篇报道的日期。
- 评估提示：S-1 的“上半年约 48% 收入来自云市场”只能用来打分，不能作为输入。可以拿来检验 Bedrock、Foundry 增速这类方向信号，事后看是否和云渠道收入的变化一致。
