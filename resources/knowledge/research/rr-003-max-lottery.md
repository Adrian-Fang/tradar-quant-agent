---
research_id: RR-003
date: 2026-08-28
market: A-share
topic: MAX lottery-demand anomaly
strategy: MAX20 score = -MAX20
status: inconclusive
tags: [MAX, lottery-demand, reversal, volatility, residual]
source_type: slack
source_ref: C0BTC7LU07L/1787913920.876949
artifacts: [data/factor_defs/max_score.yaml, scripts/stocks_active/max_lottery.py]
supersedes: []
---

# Research Question

过去 20 个交易日最大单日 adjusted close-to-close return 是否带来下一月较低收益，且不是普通短期反转或低波动暴露的替代表达。

# Method

2021-01-01 至 2026-07-31，月末截面、H20。`MAX20` 为 rolling 20 日最大日收益，研究 score 为 `-MAX20`；标准 eligible universe。比较 raw、控制 `NORMAL_M` 的 `RESID_REV`，以及再控制 `VOL20`、log market cap、log ADV20 的 `RESID_FULL`，并做 VOL20 tercile 条件排序。

# Key Findings

- Raw MAX_SCORE FULL MeanIC / ICIR / PosIC `+9.84%/0.75/76.1%`；DEV_IS `+11.42%/1.14/86.1%`；DEV_OOS `+8.45%/0.68/70.8%`；POST_2026 `+6.44%/0.25/42.9%`。
- Raw FULL 五组均值从 G1 到 G5 为 `0.41%, 1.24%, 1.53%, 1.79%, 1.81%`，与 score 方向一致但不是强单调；十组 spread `+1.24%`。
- 控制 reversal 后 RESID_REV FULL MeanIC `+6.72%`；加入 volatility/size/liquidity 后 RESID_FULL FULL 仅 `+0.41%`，DEV_OOS `-1.42%`，PosIC `47.8%`。
- RESID_FULL FULL 五组 spread `+0.12%`，十组 G10-G1 `-0.36%`；高波动 tercile FULL IC/spread `-3.04%/-0.65%`，低波动 tercile `+2.63%/+0.59%`。
- 结果表明 raw anomaly 很大程度混合了低波动和反转 exposure；残差后的独立信息很弱且不稳定。

# Conclusion

`inconclusive`：MAX raw 结构存在，但当前证据不足以把它视为独立 MAX alpha。若继续，只能先明确地拆解 volatility/reversal，不应直接优化组合。

# Caveats

样本是 67 个完整月末；RESID_FULL 的正向结果主要不稳定于 OOS/高波动子样本。没有 production portfolio 结论。

# Provenance

Slack `#tradar-research`：max-lottery-001 线程 `1787913920.876949`。
