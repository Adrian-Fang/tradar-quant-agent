---
research_id: RR-001
date: 2026-08-31
market: A-share
topic: Relative strength versus Shanghai Composite and short-term reversal
strategy: RS_Pct252 × Excess20 structure
status: inconclusive
tags: [relative-strength, short-term-reversal, cross-sectional, factor]
source_type: slack
source_ref: [C0BTC7LU07L/1788162013.552609, C0BTC7LU07L/1788162447.857919]
artifacts: [scripts/relative_momentum/relative_strength_vs_index.py]
supersedes: []
---

# Research Question

是否 `RS_Pct252`（个股/上证指数相对强弱在自身 252 日历史分位）在控制 `Excess20` 后仍有稳定信息，还是主要由短期相对反转解释。

# Method

2021-01-01 至 2026-07-31，日频股票截面，另做固定每 5 个交易日 snapshot。股票与上证指数使用 repo 前复权 close；基础过滤排除 ST、252 日窗口不足和无效价格。计算 `RS`、`RS_Pct252`、`Excess20`、`RS_Change20`，未来 H5/H10/H20 只作评价。

# Key Findings

- `corr(Excess20, RS_Change20)`：DAILY FULL `+0.9988`；DEV_IS `+0.9993`；DEV_OOS `+0.9984`；2026YTD `+0.9994`。两者高度冗余，未合成综合分数。
- FULL DAILY 5×5 H20（绝对/对上证超额）：A 高 RS+高 Excess `+0.23%/-0.07%`；B 高 RS+低 Excess `+1.29%/+0.92%`；C 低/中 RS+高 Excess `+0.88%/+0.51%`；D 低/中 RS+低 Excess `+1.47%/+1.16%`。
- 阶段方向不稳：D 在 DEV_IS `+0.64%/+1.24%`、DEV_OOS `+3.56%/+1.94%`，但 2026YTD `-3.02%/-1.97%`；A 在 DEV_IS 和 2026YTD 绝对收益为负。
- Excess20 单因子 FULL H20 Q1→Q5 为 `+1.38%, +1.48%, +1.36%, +1.04%, -0.09%`，呈短期相对反转；2026YTD 方向反转。RS_Pct252 FULL 为 `+1.27%, +1.50%, +1.21%, +0.89%, +0.31%`。
- Excess quintile 内 RS 高低增量不稳定；FULL DAILY Q5 的 `RS Q5-Q1` 超额差 `+1.48%` 主要由 2026YTD 极少量样本贡献，Q1-Q4 接近零或为负；WEEKLY FULL 同样只在 Q5 为 `+1.85%`。

# Conclusion

`inconclusive`：可重复观察到的是 Excess20 的短期相对反转，不是稳定的 252 日相对强弱独立 alpha。`RS_INCREMENTAL=false`；不应把该 5×5 结构直接升级为策略。

# Caveats

日频样本高度重叠；周频方向大致一致但不能消除 2026 小样本问题。相关性接近 1 也意味着 `Excess20` 与 `RS_Change20` 不是两个独立暴露。

# Provenance

Slack `#tradar-research`：relative-strength-vs-index-002 线程 `1788162013.552609`；relative-reversal-decompose-001 线程 `1788162447.857919`。
